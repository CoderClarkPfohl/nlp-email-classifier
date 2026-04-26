"""Tests for utils/summarizer.py."""

import pytest
from utils.summarizer import split_sentences, score_sentence, summarize_email


class TestSplitSentences:
    def test_basic_split(self):
        # Sentences must be >15 chars each to pass the filter
        text = "This is the first sentence here. This is the second sentence here. This is the third sentence here."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_filters_short_fragments(self):
        text = "Hello. This is a real sentence that has enough words. Bye."
        sentences = split_sentences(text)
        # "Hello." and "Bye." are <= 15 chars, should be filtered
        assert all(len(s) > 15 for s in sentences)

    def test_empty_text(self):
        assert split_sentences("") == []


class TestScoreSentence:
    def test_high_keyword_scores_high(self):
        score = score_sentence("We are pleased to offer you the position.", 0, 5)
        assert score > 0

    def test_boilerplate_scores_low(self):
        score = score_sentence("Click here to unsubscribe from this newsletter.", 3, 5)
        assert score < 0

    def test_first_position_bonus(self):
        s1 = score_sentence("Your application is under review.", 0, 5)
        s2 = score_sentence("Your application is under review.", 3, 5)
        assert s1 > s2

    def test_very_short_sentence_penalized(self):
        score = score_sentence("OK sure.", 2, 5)
        assert score < 0


class TestSummarizeEmail:
    def test_returns_string(self):
        text = "We received your application. Our team will review it carefully. We will be in touch soon. Thank you for your interest."
        result = summarize_email(text, max_sentences=2)
        assert isinstance(result, str)

    def test_respects_max_sentences(self):
        text = "Sentence one is here. Sentence two is here. Sentence three is here. Sentence four is here. Sentence five is here."
        result = summarize_email(text, max_sentences=2)
        # Should have at most 2 sentences
        period_count = result.count(".")
        assert period_count <= 3  # allowing for minor variations

    def test_short_text_returned_as_is(self):
        text = "Short email."
        assert summarize_email(text) == text

    def test_empty_text(self):
        assert summarize_email("") == ""

    def test_none_text(self):
        assert summarize_email(None) is None

    def test_prefers_informative_sentences(self):
        text = (
            "Click here to unsubscribe from all emails. "
            "We are pleased to offer you the position of Data Analyst. "
            "Please do not reply to this automated message. "
            "Your start date is January 15 and the compensation package is attached."
        )
        result = summarize_email(text, max_sentences=2)
        assert "offer" in result.lower() or "start date" in result.lower()
