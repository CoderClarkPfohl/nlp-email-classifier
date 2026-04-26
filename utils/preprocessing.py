"""
Text preprocessing for job application emails.
Handles HTML removal, normalization, tokenization, stop word removal,
and morphological stemming (Topic 1: Basic Text Processing).
"""

import re
from typing import List


# Common English stop words (subset sufficient for TF-IDF context)
STOP_WORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "same", "so", "than", "too", "very", "just", "if",
    "about", "up", "out", "then", "here", "there", "am", "as", "also",
])


def strip_html(text: str) -> str:
    """Remove HTML tags, entities, and excessive whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#?\w+;", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_email_addresses(text: str) -> str:
    """Remove email addresses (but preserve the surrounding text)."""
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def clean_email_body(text: str) -> str:
    """
    Full cleaning pipeline for a raw email body.
    Returns lowercased, cleaned text ready for feature extraction.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = strip_html(text)
    text = remove_urls(text)
    text = remove_email_addresses(text)
    # Remove non-alphanumeric characters except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", " ", text)
    text = normalize_whitespace(text)
    return text.lower()


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out stop words from a token list."""
    return [t for t in tokens if t not in STOP_WORDS]


# ─────────────────────────────────────────────────────────────
#  Suffix-stripping stemmer (Topic 1: morphological processing)
# ─────────────────────────────────────────────────────────────

# Rules ordered by specificity (longer suffixes checked first).
# Each rule: (suffix, minimum stem length after removal, replacement)
_STEM_RULES = [
    # Inflectional morphology — verb forms
    ("ying", 2, "y"),   # applying -> apply
    ("ied", 2, "y"),    # applied -> apply
    ("ing", 3, ""),     # reviewing -> review
    ("ed", 3, ""),      # received -> receiv
    # Inflectional morphology — plurals
    ("sses", 2, "ss"),  # addresses -> address
    ("ies", 2, "y"),    # companies -> company
    # Derivational morphology — nominal suffixes
    ("ation", 3, ""),   # application -> applic
    ("ment", 3, ""),    # assessment -> assess
    ("ness", 3, ""),    # weakness -> weak
    ("ence", 3, ""),    # experience -> experi
    ("ance", 3, ""),    # performance -> perform
    # Derivational morphology — adjectival suffixes
    ("able", 3, ""),    # available -> avail
    ("ible", 3, ""),    # possible -> poss
    ("ful", 3, ""),     # successful -> success
    ("ous", 3, ""),     # previous -> previ
    ("ive", 3, ""),     # competitive -> competit
    # Adverb
    ("ally", 3, "al"),  # technically -> technical
    ("ly", 3, ""),      # unfortunately -> unfortunat
    # Plural (must be last — least specific)
    ("s", 3, ""),       # candidates -> candidate
]


def stem_word(word: str) -> str:
    """
    Apply suffix-stripping rules to reduce a word to its stem.

    Demonstrates morphological processing (Topic 1: Basic Text Processing).
    Stemming normalizes inflected and derived forms to a common base,
    reducing vocabulary size and helping TF-IDF capture semantic
    similarity between forms like 'apply', 'applied', 'applying'.

    Uses a rule-based approach inspired by the Porter stemmer, handling
    the most impactful English suffix patterns for this domain.
    """
    if len(word) <= 3:
        return word

    for suffix, min_stem, replacement in _STEM_RULES:
        if word.endswith(suffix):
            stem = word[: -len(suffix)] + replacement
            if len(stem) >= min_stem:
                return stem

    return word


# Regex matching sklearn's default token_pattern for consistency
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def stemming_tokenizer(text: str) -> List[str]:
    """
    Tokenize and stem text in one step.

    Used as a custom tokenizer for TfidfVectorizer to incorporate
    morphological normalization into the TF-IDF feature space.
    Matches sklearn's default tokenization behavior, adding only stemming.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [stem_word(t) for t in tokens]


def preprocess_for_model(text: str) -> str:
    """
    Light cleaning intended for transformer models that benefit from
    more natural text (DeBERTa, BERT, etc.).
    Keeps casing and punctuation but removes HTML and noise.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = strip_html(text)
    text = remove_urls(text)
    text = normalize_whitespace(text)
    # Truncate to ~512 tokens worth of text (rough char estimate)
    if len(text) > 2000:
        text = text[:2000]
    return text
