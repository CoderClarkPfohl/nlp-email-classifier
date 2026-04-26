"""Tests for utils/preprocessing.py."""

import pytest
from utils.preprocessing import (
    strip_html, normalize_whitespace, remove_urls, remove_email_addresses,
    clean_email_body, tokenize, remove_stopwords, preprocess_for_model,
    stem_word, stemming_tokenizer,
)


class TestStripHtml:
    def test_removes_tags(self):
        assert "hello" in strip_html("<p>hello</p>")

    def test_removes_entities(self):
        assert "&amp;" not in strip_html("a &amp; b")

    def test_plain_text_unchanged(self):
        assert strip_html("no html here").strip() == "no html here"


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("a   b") == "a b"

    def test_collapses_newlines(self):
        assert normalize_whitespace("a\n\n\nb") == "a b"

    def test_strips_edges(self):
        assert normalize_whitespace("  hi  ") == "hi"


class TestRemoveUrls:
    def test_removes_http(self):
        assert "click" in remove_urls("click http://example.com here")
        assert "http" not in remove_urls("click http://example.com here")

    def test_removes_https(self):
        assert "https" not in remove_urls("visit https://foo.com/bar")


class TestRemoveEmailAddresses:
    def test_removes_email(self):
        result = remove_email_addresses("contact us at hr@company.com today")
        assert "hr@company.com" not in result
        assert "contact" in result


class TestCleanEmailBody:
    def test_full_pipeline(self):
        html = "<p>Visit https://x.com or email hr@x.com — Thank You!</p>"
        result = clean_email_body(html)
        assert "<p>" not in result
        assert "https" not in result
        assert "hr@x.com" not in result
        assert result == result.lower()

    def test_empty_string(self):
        assert clean_email_body("") == ""

    def test_none_returns_empty(self):
        assert clean_email_body(None) == ""

    def test_whitespace_only(self):
        assert clean_email_body("   ") == ""


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("Hello world, this is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens

    def test_filters_short_tokens(self):
        tokens = tokenize("I am a person")
        assert "i" not in tokens  # single char filtered
        assert "am" in tokens


class TestRemoveStopwords:
    def test_removes_common_stopwords(self):
        tokens = ["the", "cat", "is", "on", "the", "mat"]
        result = remove_stopwords(tokens)
        assert "cat" in result
        assert "mat" in result
        assert "the" not in result
        assert "is" not in result


class TestStemWord:
    def test_short_words_unchanged(self):
        assert stem_word("the") == "the"
        assert stem_word("is") == "is"
        assert stem_word("at") == "at"

    def test_plural_s(self):
        assert stem_word("candidates") == "candidate"

    def test_plural_ies(self):
        assert stem_word("companies") == "company"

    def test_plural_sses(self):
        assert stem_word("addresses") == "address"

    def test_ing_removal(self):
        assert stem_word("reviewing") == "review"

    def test_ed_removal(self):
        assert stem_word("received") == "receiv"

    def test_ied_to_y(self):
        assert stem_word("applied") == "apply"

    def test_ying_to_y(self):
        assert stem_word("applying") == "apply"

    def test_derivational_ation(self):
        assert stem_word("application") == "applic"

    def test_derivational_ment(self):
        assert stem_word("assessment") == "assess"

    def test_derivational_ness(self):
        assert stem_word("weakness") == "weak"

    def test_derivational_ence(self):
        assert stem_word("experience") == "experi"

    def test_derivational_able(self):
        assert stem_word("available") == "avail"

    def test_word_too_short_for_rule(self):
        """Stem must meet min_stem threshold."""
        # "us" + suffix "s" — stem would be "u" (len 1) < min_stem 3
        assert stem_word("us") == "us"

    def test_preserves_unrecognized_words(self):
        assert stem_word("python") == "python"


class TestStemmingTokenizer:
    def test_returns_list_of_strings(self):
        result = stemming_tokenizer("the quick brown fox")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_stems_are_applied(self):
        result = stemming_tokenizer("application reviewing weakness")
        assert "applic" in result   # -ation removal
        assert "review" in result   # -ing removal
        assert "weak" in result     # -ness removal

    def test_lowercases(self):
        result = stemming_tokenizer("Hello World")
        assert all(t == t.lower() for t in result)


class TestPreprocessForModel:
    def test_preserves_casing(self):
        result = preprocess_for_model("Hello World")
        assert "Hello" in result

    def test_removes_html(self):
        result = preprocess_for_model("<b>Bold</b> text")
        assert "<b>" not in result
        assert "Bold" in result

    def test_truncates_long_text(self):
        long = "a" * 5000
        result = preprocess_for_model(long)
        assert len(result) <= 2000

    def test_empty_input(self):
        assert preprocess_for_model("") == ""
        assert preprocess_for_model(None) == ""
