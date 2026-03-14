"""
Extractive summarization for job application emails.
Uses sentence scoring based on keyword relevance, position, and length
to select the most informative sentences.
"""

import re
from typing import List


# Keywords that signal important content in job application emails
IMPORTANCE_KEYWORDS = {
    "high": [
        "offer", "congratulations", "accepted", "interview", "schedule",
        "assessment", "rejected", "unfortunately", "regret", "not moving",
        "next steps", "deadline", "complete", "action required",
    ],
    "medium": [
        "application", "received", "review", "position", "role",
        "team", "status", "update", "candidate", "process",
    ],
}


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def score_sentence(sentence: str, position: int, total: int) -> float:
    """
    Score a sentence for inclusion in the summary.

    Factors:
    - Presence of high/medium importance keywords
    - Position bonus (first and last sentences are often important)
    - Length penalty (too short = noise, too long = boilerplate)
    """
    score = 0.0
    lower = sentence.lower()

    # Keyword scoring
    for kw in IMPORTANCE_KEYWORDS["high"]:
        if kw in lower:
            score += 3.0
    for kw in IMPORTANCE_KEYWORDS["medium"]:
        if kw in lower:
            score += 1.0

    # Position bonus: first few sentences and last sentence
    if position <= 1:
        score += 2.0
    elif position == total - 1:
        score += 1.0

    # Length: prefer medium-length sentences
    word_count = len(sentence.split())
    if 8 <= word_count <= 30:
        score += 1.0
    elif word_count < 5:
        score -= 2.0

    # Penalty for boilerplate patterns
    boilerplate = [
        "unsubscribe", "privacy policy", "click here", "view in browser",
        "all rights reserved", "copyright", "terms of use",
        "please do not reply", "this email was sent",
    ]
    for bp in boilerplate:
        if bp in lower:
            score -= 5.0

    return score


def summarize_email(text: str, max_sentences: int = 3) -> str:
    """
    Produce an extractive summary of a job application email.

    Parameters
    ----------
    text : str
        Cleaned email body text.
    max_sentences : int
        Maximum number of sentences in the summary.

    Returns
    -------
    str : The summary composed of top-scored sentences in original order.
    """
    if not text or len(text) < 30:
        return text

    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    scored = []
    for i, sent in enumerate(sentences):
        s = score_sentence(sent, i, len(sentences))
        scored.append((i, s, sent))

    # Pick top sentences, preserve original order
    scored.sort(key=lambda x: x[1], reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[0])
    return " ".join(t[2] for t in top)
