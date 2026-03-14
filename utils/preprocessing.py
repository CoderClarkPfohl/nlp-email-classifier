"""
Text preprocessing for job application emails.
Handles HTML removal, normalization, tokenization, and stop word removal.
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
