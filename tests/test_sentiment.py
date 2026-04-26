"""Tests for models/sentiment.py."""

import pytest
from models.sentiment import compute_sentiment


class TestPositiveSentiment:
    def test_offer_email(self):
        result = compute_sentiment("Congratulations! We are pleased to offer you the position.")
        assert result["label"] == "positive"
        assert result["compound"] > 0

    def test_excited_language(self):
        result = compute_sentiment("We are excited and thrilled to welcome you aboard!")
        assert result["label"] == "positive"


class TestNegativeSentiment:
    def test_rejection_email(self):
        result = compute_sentiment("Unfortunately, we regret to inform you that you were not selected.")
        assert result["label"] == "negative"
        assert result["compound"] < 0

    def test_not_moving_forward(self):
        result = compute_sentiment("We are not moving forward with your candidacy.")
        assert result["label"] == "negative"


class TestNeutralSentiment:
    def test_application_received(self):
        result = compute_sentiment("We have received your application and it is under review.")
        assert result["label"] == "neutral"

    def test_empty_text(self):
        result = compute_sentiment("")
        assert result["label"] == "neutral"
        assert result["compound"] == 0


class TestOutputStructure:
    def test_returns_all_keys(self):
        result = compute_sentiment("Test email text here.")
        expected_keys = {"positive", "negative", "neutral", "compound", "label"}
        assert set(result.keys()) == expected_keys

    def test_scores_are_numeric(self):
        result = compute_sentiment("Good news about your application.")
        assert isinstance(result["positive"], (int, float))
        assert isinstance(result["negative"], (int, float))
        assert isinstance(result["compound"], (int, float))

    def test_compound_in_range(self):
        for text in ["Great!", "Terrible!", "Neutral update."]:
            result = compute_sentiment(text)
            assert -1 <= result["compound"] <= 1

    def test_label_is_valid(self):
        for text in ["Congratulations!", "Unfortunately declined.", "Under review."]:
            result = compute_sentiment(text)
            assert result["label"] in ("positive", "negative", "neutral")
