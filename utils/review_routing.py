"""Confidence-based routing for classifier outputs."""

import numpy as np


def assign_review_status(
    confidence,
    top2_gap,
    auto_accept_confidence: float = 0.92,
    auto_accept_gap: float = 0.35,
    low_confidence_threshold: float = 0.65,
    low_gap_threshold: float = 0.15,
) -> str:
    """Map confidence signals to a human-review status."""
    confidence = float(confidence or 0.0)
    top2_gap = float(top2_gap or 0.0)

    if confidence < low_confidence_threshold or top2_gap < low_gap_threshold:
        return "low_confidence"
    if confidence >= auto_accept_confidence and top2_gap >= auto_accept_gap:
        return "auto_accept"
    return "needs_review"


def review_statuses_from_probas(probas: np.ndarray) -> list:
    """Compute review statuses from an n_samples x n_classes probability matrix."""
    if probas.size == 0:
        return []
    sorted_probas = np.sort(probas, axis=1)
    confidence = sorted_probas[:, -1]
    if probas.shape[1] == 1:
        gaps = confidence
    else:
        gaps = sorted_probas[:, -1] - sorted_probas[:, -2]
    return [
        assign_review_status(conf, gap)
        for conf, gap in zip(confidence, gaps)
    ]
