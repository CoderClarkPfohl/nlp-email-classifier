"""
Tests for models/ngram_lm.py — per-category n-gram language models.

Covers:
- NgramLanguageModel: fit, vocab, bigram counts, log_prob, perplexity
- CategoryLanguageModels: fit, perplexity_features shape, discrimination
"""

import math
import pytest
import numpy as np

from models.ngram_lm import NgramLanguageModel, CategoryLanguageModels


# ─────────────────────────────────────────────────────────────
#  NgramLanguageModel
# ─────────────────────────────────────────────────────────────


class TestNgramLanguageModel:
    def test_fit_builds_vocab(self):
        lm = NgramLanguageModel(n=2, k=0.1)
        lm.fit([["hello", "world"], ["hello", "there"]])
        assert "hello" in lm.vocab
        assert "world" in lm.vocab
        assert "<s>" in lm.vocab
        assert "</s>" in lm.vocab

    def test_fit_builds_bigram_counts(self):
        lm = NgramLanguageModel(n=2, k=0.1)
        lm.fit([["a", "b", "c"]])
        # Bigrams: <s>->a, a->b, b->c, c-></s>
        assert lm.ngram_counts[("<s>",)]["a"] == 1
        assert lm.ngram_counts[("a",)]["b"] == 1
        assert lm.ngram_counts[("b",)]["c"] == 1
        assert lm.ngram_counts[("c",)]["</s>"] == 1

    def test_log_probability_is_negative(self):
        lm = NgramLanguageModel(n=2, k=0.1)
        lm.fit([["the", "cat", "sat"]])
        log_prob, count = lm.log_prob_sequence(["the", "cat"])
        assert log_prob < 0  # log probs are always negative
        assert count == 3  # the, cat, </s>

    def test_seen_sequence_higher_prob_than_unseen(self):
        lm = NgramLanguageModel(n=2, k=0.1)
        lm.fit([["the", "cat", "sat"]] * 10)
        lp_seen, _ = lm.log_prob_sequence(["the", "cat", "sat"])
        lp_unseen, _ = lm.log_prob_sequence(["dog", "ran", "fast"])
        assert lp_seen > lp_unseen

    def test_perplexity_lower_for_matching_text(self):
        lm = NgramLanguageModel(n=2, k=0.1)
        lm.fit([["we", "regret", "to", "inform", "you"]] * 20)
        pp_match = lm.perplexity(["we", "regret", "to", "inform"])
        pp_diff = lm.perplexity(["congratulations", "welcome", "aboard"])
        assert pp_match < pp_diff

    def test_perplexity_empty_tokens_is_inf(self):
        lm = NgramLanguageModel(n=2, k=0.1)
        lm.fit([["hello"]])
        assert lm.perplexity([]) == float("inf")

    def test_smoothing_handles_unseen_bigrams(self):
        lm = NgramLanguageModel(n=2, k=0.1)
        lm.fit([["a", "b"]])
        # "x y" was never seen, but smoothing gives nonzero prob
        pp = lm.perplexity(["x", "y"])
        assert pp > 0 and pp != float("inf")

    def test_higher_k_means_more_uniform(self):
        """More smoothing -> perplexity closer between seen/unseen."""
        data = [["the", "cat", "sat"]] * 20
        lm_low = NgramLanguageModel(n=2, k=0.01)
        lm_low.fit(data)
        lm_high = NgramLanguageModel(n=2, k=1.0)
        lm_high.fit(data)

        seen = ["the", "cat", "sat"]
        unseen = ["dog", "ran", "fast"]

        gap_low = lm_low.perplexity(unseen) - lm_low.perplexity(seen)
        gap_high = lm_high.perplexity(unseen) - lm_high.perplexity(seen)
        # More smoothing -> smaller gap between seen and unseen
        assert gap_high < gap_low

    def test_trigram_model(self):
        lm = NgramLanguageModel(n=3, k=0.1)
        lm.fit([["a", "b", "c", "d"]])
        pp = lm.perplexity(["a", "b", "c"])
        assert pp > 0 and pp != float("inf")

    def test_perplexity_is_positive(self):
        lm = NgramLanguageModel(n=2, k=0.1)
        lm.fit([["hello", "world"]])
        assert lm.perplexity(["hello", "world"]) > 0

    def test_repeated_training_data_lowers_perplexity(self):
        """More training data on same pattern -> lower perplexity."""
        data_small = [["a", "b", "c"]] * 2
        data_large = [["a", "b", "c"]] * 100
        lm_s = NgramLanguageModel(n=2, k=0.1)
        lm_l = NgramLanguageModel(n=2, k=0.1)
        lm_s.fit(data_small)
        lm_l.fit(data_large)
        assert lm_l.perplexity(["a", "b", "c"]) <= lm_s.perplexity(["a", "b", "c"])


# ─────────────────────────────────────────────────────────────
#  CategoryLanguageModels
# ─────────────────────────────────────────────────────────────


class TestCategoryLanguageModels:
    @pytest.fixture
    def cat_data(self):
        rejection_texts = [
            ["we", "regret", "to", "inform", "you", "not", "moving", "forward"],
            ["unfortunately", "we", "cannot", "offer", "you", "the", "position"],
            ["we", "regret", "not", "selected", "other", "candidates"],
            ["regret", "to", "inform", "decided", "not", "to", "proceed"],
            ["not", "moving", "forward", "we", "regret", "the", "decision"],
        ]
        acceptance_texts = [
            ["pleased", "to", "offer", "you", "the", "position", "welcome"],
            ["congratulations", "we", "are", "excited", "to", "extend", "offer"],
            ["welcome", "aboard", "your", "start", "date", "is", "monday"],
            ["thrilled", "to", "welcome", "you", "offer", "letter", "attached"],
            ["pleased", "to", "extend", "offer", "compensation", "package"],
        ]
        texts = rejection_texts + acceptance_texts
        labels = ["rejection"] * 5 + ["acceptance"] * 5
        return texts, labels

    def test_fit_creates_models_per_category(self, cat_data):
        texts, labels = cat_data
        clm = CategoryLanguageModels(n=2, k=0.1)
        clm.fit(texts, labels)
        assert "rejection" in clm.models
        assert "acceptance" in clm.models
        assert len(clm.categories) == 2

    def test_perplexity_features_shape(self, cat_data):
        texts, labels = cat_data
        clm = CategoryLanguageModels(n=2, k=0.1)
        clm.fit(texts, labels)
        feats = clm.perplexity_features(texts)
        assert feats.shape == (10, 2)

    def test_perplexity_features_all_positive(self, cat_data):
        texts, labels = cat_data
        clm = CategoryLanguageModels(n=2, k=0.1)
        clm.fit(texts, labels)
        feats = clm.perplexity_features(texts)
        assert np.all(feats > 0)

    def test_rejection_text_lower_pp_under_rejection_lm(self, cat_data):
        texts, labels = cat_data
        clm = CategoryLanguageModels(n=2, k=0.1)
        clm.fit(texts, labels)
        test_text = [["we", "regret", "to", "inform"]]
        feats = clm.perplexity_features(test_text)
        # Column order is alphabetical: [acceptance, rejection]
        acceptance_pp = feats[0, 0]
        rejection_pp = feats[0, 1]
        assert rejection_pp < acceptance_pp

    def test_acceptance_text_lower_pp_under_acceptance_lm(self, cat_data):
        texts, labels = cat_data
        clm = CategoryLanguageModels(n=2, k=0.1)
        clm.fit(texts, labels)
        test_text = [["pleased", "to", "offer", "welcome"]]
        feats = clm.perplexity_features(test_text)
        acceptance_pp = feats[0, 0]
        rejection_pp = feats[0, 1]
        assert acceptance_pp < rejection_pp

    def test_handles_single_category(self):
        texts = [["hello", "world"]] * 5
        labels = ["only_cat"] * 5
        clm = CategoryLanguageModels(n=2, k=0.1)
        clm.fit(texts, labels)
        feats = clm.perplexity_features(texts)
        assert feats.shape == (5, 1)

    def test_handles_many_categories(self):
        texts = [["a", "b"]] * 3 + [["c", "d"]] * 3 + [["e", "f"]] * 3
        labels = ["x"] * 3 + ["y"] * 3 + ["z"] * 3
        clm = CategoryLanguageModels(n=2, k=0.1)
        clm.fit(texts, labels)
        feats = clm.perplexity_features(texts)
        assert feats.shape == (9, 3)
        assert np.all(feats > 0)
