import numpy as np

from utils.review_routing import assign_review_status, review_statuses_from_probas


def test_assign_review_status_auto_accept():
    assert assign_review_status(0.95, 0.40) == "auto_accept"


def test_assign_review_status_low_confidence():
    assert assign_review_status(0.40, 0.30) == "low_confidence"


def test_assign_review_status_low_gap():
    assert assign_review_status(0.90, 0.02) == "low_confidence"


def test_assign_review_status_needs_review():
    assert assign_review_status(0.80, 0.20) == "needs_review"


def test_review_statuses_from_probas():
    probas = np.array(
        [
            [0.95, 0.05],
            [0.52, 0.48],
            [0.80, 0.20],
        ]
    )
    assert review_statuses_from_probas(probas) == [
        "auto_accept",
        "low_confidence",
        "needs_review",
    ]
