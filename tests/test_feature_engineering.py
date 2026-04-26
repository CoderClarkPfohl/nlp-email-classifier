"""
Tests for models/feature_engineering.py — enhanced feature engineering.

Covers:
- compute_keyword_features: shape, known matches, non-negative
- compute_ner_features: shape, entity detection, date counts
- compute_category_centroids + compute_cosine_features: shape, similarity range
- compute_sentiment_features: shape, encoding, non-negative
- compute_enhanced_features: full integration, optional columns
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from models.feature_engineering import (
    compute_keyword_features,
    KEYWORD_CATEGORIES,
    compute_ner_features,
    compute_category_centroids,
    compute_cosine_features,
    compute_sentiment_features,
    compute_enhanced_features,
)


# ─────────────────────────────────────────────────────────────
#  Keyword features (Topic 5)
# ─────────────────────────────────────────────────────────────


class TestKeywordFeatures:
    def test_shape(self):
        texts = ["hello world", "test email"]
        feats = compute_keyword_features(texts)
        assert feats.shape == (2, 6)

    def test_rejection_text_has_rejection_count(self):
        texts = ["we regret to inform you not moving forward"]
        feats = compute_keyword_features(texts)
        rej_idx = KEYWORD_CATEGORIES.index("rejection")
        assert feats[0, rej_idx] > 0

    def test_acceptance_text_has_acceptance_count(self):
        texts = ["pleased to offer you the position welcome aboard"]
        feats = compute_keyword_features(texts)
        acc_idx = KEYWORD_CATEGORIES.index("acceptance")
        assert feats[0, acc_idx] > 0

    def test_generic_text_has_zero_counts(self):
        texts = ["the quick brown fox jumps over the lazy dog"]
        feats = compute_keyword_features(texts)
        assert feats.sum() == 0

    def test_all_non_negative(self):
        texts = ["regret to inform", "pleased to offer", "schedule an interview"]
        feats = compute_keyword_features(texts)
        assert np.all(feats >= 0)

    def test_multiple_matches_counted(self):
        texts = ["regret to inform not moving forward not been selected"]
        feats = compute_keyword_features(texts)
        rej_idx = KEYWORD_CATEGORIES.index("rejection")
        assert feats[0, rej_idx] >= 3  # three strong rejection phrases


# ─────────────────────────────────────────────────────────────
#  NER features (Topic 6)
# ─────────────────────────────────────────────────────────────


class TestNerFeatures:
    def test_shape(self):
        feats = compute_ner_features(
            [None, "Engineer", None],
            [None, None, "Jane Smith"],
            [None, "jane@co.com", None],
            ["", "Mar 1; Apr 2", ""],
        )
        assert feats.shape == (3, 4)

    def test_has_role_detected(self):
        feats = compute_ner_features(
            ["Software Engineer"],
            [None],
            [None],
            [""],
        )
        assert feats[0, 0] == 1.0

    def test_date_count(self):
        feats = compute_ner_features(
            [None],
            [None],
            [None],
            ["Jan 1; Feb 2; Mar 3"],
        )
        assert feats[0, 3] == 3.0

    def test_no_entities_all_zero(self):
        feats = compute_ner_features([None], [None], [None], [""])
        assert feats.sum() == 0

    def test_all_non_negative(self):
        feats = compute_ner_features(
            ["role", None],
            ["person", None],
            ["a@b.com", None],
            ["Jan 1", ""],
        )
        assert np.all(feats >= 0)

    def test_all_entities_present(self):
        feats = compute_ner_features(
            ["Engineer"],
            ["John Doe"],
            ["j@co.com"],
            ["Jan 1; Feb 2"],
        )
        assert feats[0, 0] == 1.0  # role
        assert feats[0, 1] == 1.0  # person
        assert feats[0, 2] == 1.0  # email
        assert feats[0, 3] == 2.0  # dates


# ─────────────────────────────────────────────────────────────
#  Cosine similarity features (Topics 2 & 4)
# ─────────────────────────────────────────────────────────────


class TestCosineFeatures:
    def test_centroids_returns_dict_and_categories(self):
        X = csr_matrix(np.array([[1, 0], [0, 1], [1, 1], [0, 1]]))
        labels = ["a", "a", "b", "b"]
        centroids, cats = compute_category_centroids(X, labels)
        assert len(centroids) == 2
        assert cats == ["a", "b"]

    def test_cosine_features_shape(self):
        X = csr_matrix(np.array([[1, 0], [0, 1], [1, 1]]))
        centroids = {"a": np.array([1, 0]), "b": np.array([0, 1])}
        feats = compute_cosine_features(X, centroids, ["a", "b"])
        assert feats.shape == (3, 2)

    def test_cosine_similarity_in_valid_range(self):
        X = csr_matrix(np.random.rand(10, 5))
        labels = ["x"] * 5 + ["y"] * 5
        centroids, cats = compute_category_centroids(X, labels)
        feats = compute_cosine_features(X, centroids, cats)
        assert np.all(feats >= -0.01)
        assert np.all(feats <= 1.01)

    def test_identical_vector_has_high_similarity(self):
        X = csr_matrix(np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]]))
        centroids = {"a": np.array([1, 0, 0])}
        feats = compute_cosine_features(X, centroids, ["a"])
        assert feats[0, 0] > 0.99  # identical to centroid
        assert feats[2, 0] < 0.01  # orthogonal


# ─────────────────────────────────────────────────────────────
#  Sentiment features
# ─────────────────────────────────────────────────────────────


class TestSentimentFeatures:
    def test_shape(self):
        feats = compute_sentiment_features(
            [0.5, -0.3, 0.0], ["positive", "negative", "neutral"]
        )
        assert feats.shape == (3, 4)

    def test_compound_shifted_to_non_negative(self):
        feats = compute_sentiment_features(
            [-1.0, 0.0, 1.0], ["negative", "neutral", "positive"]
        )
        assert np.all(feats[:, 0] >= 0)

    def test_positive_encoding(self):
        feats = compute_sentiment_features([0.5], ["positive"])
        assert feats[0, 1] == 1.0  # is_positive
        assert feats[0, 2] == 0.0
        assert feats[0, 3] == 0.0

    def test_negative_encoding(self):
        feats = compute_sentiment_features([-0.5], ["negative"])
        assert feats[0, 2] == 1.0  # is_negative

    def test_neutral_encoding(self):
        feats = compute_sentiment_features([0.0], ["neutral"])
        assert feats[0, 3] == 1.0  # is_neutral

    def test_all_non_negative(self):
        feats = compute_sentiment_features(
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            ["negative", "negative", "neutral", "positive", "positive"],
        )
        assert np.all(feats >= 0)


# ─────────────────────────────────────────────────────────────
#  compute_enhanced_features (full integration)
# ─────────────────────────────────────────────────────────────


class TestComputeEnhancedFeatures:
    @pytest.fixture
    def enriched_df(self):
        """DataFrame mimicking pipeline output with all required columns."""
        n = 30
        labels = (
            ["acceptance"] * 5
            + ["rejection"] * 5
            + ["interview"] * 5
            + ["action_required"] * 5
            + ["in_process"] * 5
            + ["unrelated"] * 5
        )

        bodies = []
        for label in labels:
            if label == "acceptance":
                bodies.append(
                    "pleased to offer you the position welcome aboard start date"
                )
            elif label == "rejection":
                bodies.append(
                    "regret to inform you not moving forward other candidates"
                )
            elif label == "interview":
                bodies.append(
                    "invite you for an interview phone screen schedule call"
                )
            elif label == "action_required":
                bodies.append(
                    "please complete the online assessment coding challenge deadline"
                )
            elif label == "in_process":
                bodies.append(
                    "thank you for applying received your application under review"
                )
            else:
                bodies.append(
                    "new jobs matching your profile unsubscribe job alert newsletter"
                )

        return pd.DataFrame(
            {
                "clean_body": bodies,
                "rule_label": labels,
                "extracted_role": [None] * n,
                "contact_person": [None] * n,
                "contact_email": [None] * n,
                "dates_mentioned": [""] * n,
                "sentiment_compound": [0.5] * 5
                + [-0.5] * 5
                + [0.0] * 20,
                "sentiment_label": ["positive"] * 5
                + ["negative"] * 5
                + ["neutral"] * 20,
            }
        )

    def test_output_shape(self, enriched_df):
        feats = compute_enhanced_features(enriched_df)
        assert feats.shape[0] == 30
        # 6 LM + 6 cosine + 6 keyword + 4 NER + 4 sentiment = 26
        assert feats.shape[1] == 26

    def test_all_non_negative(self, enriched_df):
        feats = compute_enhanced_features(enriched_df)
        assert np.all(feats >= 0)

    def test_without_ner_columns(self, enriched_df):
        df = enriched_df.drop(
            columns=[
                "extracted_role",
                "contact_person",
                "contact_email",
                "dates_mentioned",
            ]
        )
        feats = compute_enhanced_features(df)
        # 6 LM + 6 cosine + 6 keyword + 4 sentiment = 22
        assert feats.shape == (30, 22)

    def test_without_sentiment_columns(self, enriched_df):
        df = enriched_df.drop(columns=["sentiment_compound", "sentiment_label"])
        feats = compute_enhanced_features(df)
        # 6 LM + 6 cosine + 6 keyword + 4 NER = 22
        assert feats.shape == (30, 22)

    def test_minimal_df(self):
        """Only clean_body and rule_label -- no NER, no sentiment."""
        df = pd.DataFrame(
            {
                "clean_body": ["offer you the position"] * 5
                + ["regret to inform"] * 5,
                "rule_label": ["acceptance"] * 5 + ["rejection"] * 5,
            }
        )
        feats = compute_enhanced_features(df)
        # 2 LM + 2 cosine + 6 keyword = 10
        assert feats.shape == (10, 10)
        assert np.all(feats >= 0)

    def test_feature_values_differ_across_categories(self, enriched_df):
        """Enhanced features should differ between acceptance and rejection rows."""
        feats = compute_enhanced_features(enriched_df)
        acceptance_mean = feats[:5].mean(axis=0)
        rejection_mean = feats[5:10].mean(axis=0)
        # At least some feature columns should differ meaningfully
        diffs = np.abs(acceptance_mean - rejection_mean)
        assert diffs.max() > 0.01
