"""
Lexicon-based sentiment analysis for job application emails.
Provides positive / negative / neutral scoring without NLTK or external models.
Tuned for the language commonly found in recruiter emails.
"""

from typing import Dict

# Domain-specific sentiment lexicon for job application emails
POSITIVE_WORDS = frozenset([
    "congratulations", "pleased", "delighted", "excited", "thrilled",
    "welcome", "offer", "accepted", "selected", "progressing",
    "impressed", "excellent", "outstanding", "successful", "advancing",
    "opportunity", "great", "happy", "wonderful", "fantastic",
    "looking forward", "promising", "strong", "qualified", "fit",
    "moved forward", "shortlisted", "finalist",
])

NEGATIVE_WORDS = frozenset([
    "unfortunately", "regret", "rejected", "declined", "unsuccessful",
    "unable", "sorry", "not selected", "not moving", "decided not",
    "not proceed", "not advance", "not a fit", "not be moving",
    "competitive", "difficult decision", "other candidates",
    "will not", "cannot", "no longer", "closed", "filled",
    "not eligible", "not qualified",
])

NEUTRAL_WORDS = frozenset([
    "received", "submitted", "review", "process", "application",
    "status", "update", "pending", "evaluation", "consideration",
])


def compute_sentiment(text: str) -> Dict[str, float]:
    """
    Compute sentiment scores for job-application email text.

    Returns
    -------
    dict with keys:
        positive : float  (0-1 score)
        negative : float  (0-1 score)
        neutral  : float  (0-1 score)
        compound : float  (-1 to +1 overall sentiment)
        label    : str    ('positive', 'negative', or 'neutral')
    """
    if not text:
        return {"positive": 0, "negative": 0, "neutral": 0,
                "compound": 0, "label": "neutral"}

    lower = text.lower()
    words = lower.split()
    total = max(len(words), 1)

    pos_count = sum(1 for w in POSITIVE_WORDS if w in lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in lower)
    neu_count = sum(1 for w in NEUTRAL_WORDS if w in lower)

    # Also check multi-word phrases
    for phrase in ["not moving", "decided not", "not selected",
                   "not proceed", "not a fit", "not be moving",
                   "difficult decision", "other candidates",
                   "looking forward"]:
        if phrase in lower:
            if phrase == "looking forward":
                pos_count += 1
            else:
                neg_count += 1

    denom = max(pos_count + neg_count + neu_count, 1)
    pos_score = pos_count / denom
    neg_score = neg_count / denom
    neu_score = neu_count / denom

    # Compound: range -1 to +1
    compound = (pos_count - neg_count) / max(pos_count + neg_count, 1)

    if compound > 0.15:
        label = "positive"
    elif compound < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return {
        "positive": round(pos_score, 4),
        "negative": round(neg_score, 4),
        "neutral": round(neu_score, 4),
        "compound": round(compound, 4),
        "label": label,
    }
