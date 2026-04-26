"""
Tests for models/svm_classifier.py

Covers:
- oversample_minority: target counts reached, output shape preserved
- SoftVotingEnsemble: fit/predict/predict_proba contract
- build_tfidf / build_ensemble: factory outputs
- train_and_evaluate: return shape, metrics keys, cv_probas shape
"""

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

from models.svm_classifier import (
    oversample_minority,
    SoftVotingEnsemble,
    build_tfidf,
    build_ensemble,
    train_and_evaluate,
)


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

# 6 classes × 5 samples = 30 rows; enough for CalibratedClassifierCV(cv=3)
_BASE_CORPUS = [
    ("we are pleased to offer you congratulations welcome aboard start date", "acceptance"),
    ("we regret inform you not selected unfortunately cannot offer", "rejection"),
    ("invite you interview schedule phone screen video interview technical", "interview"),
    ("complete assessment online coding challenge hackerrank action required", "action_required"),
    ("thank you for applying application received under review will be in touch", "in_process"),
    ("unsubscribe newsletter job alert similar jobs weekly digest recommendations", "unrelated"),
]


@pytest.fixture
def simple_texts():
    """30-sample balanced corpus (5 per class) — enough for cv=3 calibration."""
    texts = []
    for body, _ in _BASE_CORPUS:
        for i in range(5):
            texts.append(f"{body} sample {i}")
    return texts


@pytest.fixture
def simple_labels():
    labels = []
    for _, label in _BASE_CORPUS:
        labels.extend([label] * 5)
    return labels


@pytest.fixture
def tfidf_matrix(simple_texts):
    tfidf = build_tfidf()
    return tfidf.fit_transform(simple_texts)


# ─────────────────────────────────────────────────────────────
#  oversample_minority
# ─────────────────────────────────────────────────────────────
class TestOversampleMinority:
    def test_minority_classes_reach_target(self, tfidf_matrix, simple_labels):
        """Every class should have at least min_samples rows after oversampling."""
        min_samples = 5
        X_os, y_os = oversample_minority(tfidf_matrix, simple_labels, min_samples=min_samples)
        from collections import Counter
        counts = Counter(y_os)
        for label, count in counts.items():
            assert count >= min_samples, f"Class '{label}' has only {count} samples"

    def test_output_lengths_match(self, tfidf_matrix, simple_labels):
        X_os, y_os = oversample_minority(tfidf_matrix, simple_labels, min_samples=5)
        assert X_os.shape[0] == len(y_os)

    def test_sparse_matrix_preserved(self, tfidf_matrix, simple_labels):
        X_os, _ = oversample_minority(tfidf_matrix, simple_labels, min_samples=5)
        assert issparse(X_os), "Output should still be a sparse matrix"

    def test_no_oversampling_needed(self, tfidf_matrix, simple_labels):
        """When min_samples=1, nothing should be added."""
        X_os, y_os = oversample_minority(tfidf_matrix, simple_labels, min_samples=1)
        assert X_os.shape[0] == len(simple_labels)

    def test_labels_are_subset_of_original(self, tfidf_matrix, simple_labels):
        _, y_os = oversample_minority(tfidf_matrix, simple_labels, min_samples=5)
        original_set = set(simple_labels)
        assert set(y_os).issubset(original_set)

    def test_feature_dim_unchanged(self, tfidf_matrix, simple_labels):
        X_os, _ = oversample_minority(tfidf_matrix, simple_labels, min_samples=5)
        assert X_os.shape[1] == tfidf_matrix.shape[1]

    def test_deterministic_with_same_seed(self, tfidf_matrix, simple_labels):
        _, y1 = oversample_minority(tfidf_matrix, simple_labels, min_samples=5, random_state=0)
        _, y2 = oversample_minority(tfidf_matrix, simple_labels, min_samples=5, random_state=0)
        assert y1 == y2


# ─────────────────────────────────────────────────────────────
#  SoftVotingEnsemble
# ─────────────────────────────────────────────────────────────
class TestSoftVotingEnsemble:
    def test_fit_sets_classes(self, tfidf_matrix, simple_labels):
        ens = build_ensemble()
        ens.fit(tfidf_matrix, simple_labels)
        assert ens.classes_ is not None
        assert set(ens.classes_) == set(simple_labels)

    def test_predict_returns_known_labels(self, tfidf_matrix, simple_labels):
        ens = build_ensemble()
        ens.fit(tfidf_matrix, simple_labels)
        preds = ens.predict(tfidf_matrix)
        assert len(preds) == tfidf_matrix.shape[0]
        assert set(preds).issubset(set(simple_labels))

    def test_predict_proba_shape(self, tfidf_matrix, simple_labels):
        ens = build_ensemble()
        ens.fit(tfidf_matrix, simple_labels)
        probas = ens.predict_proba(tfidf_matrix)
        n_classes = len(set(simple_labels))
        assert probas.shape == (tfidf_matrix.shape[0], n_classes)

    def test_predict_proba_sums_to_one(self, tfidf_matrix, simple_labels):
        ens = build_ensemble()
        ens.fit(tfidf_matrix, simple_labels)
        probas = ens.predict_proba(tfidf_matrix)
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-6)

    def test_predict_consistent_with_proba(self, tfidf_matrix, simple_labels):
        """predict() should agree with argmax of predict_proba()."""
        ens = build_ensemble()
        ens.fit(tfidf_matrix, simple_labels)
        preds = ens.predict(tfidf_matrix)
        probas = ens.predict_proba(tfidf_matrix)
        argmax_preds = ens.classes_[np.argmax(probas, axis=1)]
        np.testing.assert_array_equal(preds, argmax_preds)

    def test_weighted_vs_unweighted(self, tfidf_matrix, simple_labels):
        """Weighted ensemble should produce different probas than unweighted."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB

        lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        nb = MultinomialNB()

        ens_w = SoftVotingEnsemble(classifiers=[lr, nb], weights=[3.0, 1.0])
        ens_u = SoftVotingEnsemble(classifiers=[lr, nb], weights=None)

        # Fit both (need separate classifier instances — reuse after fit is fine here)
        ens_w.fit(tfidf_matrix, simple_labels)
        # Re-create classifiers so they're unfitted for the second ensemble
        lr2 = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        nb2 = MultinomialNB()
        ens_u.classifiers = [lr2, nb2]
        ens_u.fit(tfidf_matrix, simple_labels)

        p_w = ens_w.predict_proba(tfidf_matrix)
        p_u = ens_u.predict_proba(tfidf_matrix)
        # They should differ (different weights → different avg)
        assert not np.allclose(p_w, p_u)


# ─────────────────────────────────────────────────────────────
#  build_tfidf / build_ensemble factories
# ─────────────────────────────────────────────────────────────
class TestFactories:
    def test_build_tfidf_is_vectorizer(self):
        v = build_tfidf()
        assert isinstance(v, TfidfVectorizer)

    def test_build_tfidf_respects_max_features(self):
        v = build_tfidf(max_features=500)
        assert v.max_features == 500

    def test_build_tfidf_ngram_range(self):
        v = build_tfidf()
        assert v.ngram_range == (1, 2)

    def test_build_ensemble_returns_soft_voting(self):
        ens = build_ensemble()
        assert isinstance(ens, SoftVotingEnsemble)

    def test_build_ensemble_has_three_classifiers(self):
        ens = build_ensemble()
        assert len(ens.classifiers) == 3

    def test_build_ensemble_has_weights(self):
        ens = build_ensemble()
        assert ens.weights is not None
        assert len(ens.weights) == 3


# ─────────────────────────────────────────────────────────────
#  train_and_evaluate
# ─────────────────────────────────────────────────────────────
class TestTrainAndEvaluate:
    """
    These tests use a larger synthetic corpus so stratified 5-fold CV
    actually has enough samples per class.
    """

    @pytest.fixture
    def corpus(self):
        base = [
            ("we are pleased to offer you congratulations welcome aboard start date", "acceptance"),
            ("we regret inform you not selected unfortunately cannot offer", "rejection"),
            ("invite you interview schedule phone screen video interview technical", "interview"),
            ("complete assessment online coding challenge hackerrank action required", "action_required"),
            ("thank you for applying application received under review will be in touch", "in_process"),
            ("unsubscribe newsletter job alert similar jobs weekly digest job recommendations", "unrelated"),
        ]
        texts, labels = [], []
        for body, label in base:
            for i in range(10):  # 10 copies each → 60 total, 10 per class
                texts.append(f"{body} variant {i}")
                labels.append(label)
        return texts, labels

    def test_returns_five_tuple(self, corpus):
        texts, labels = corpus
        result = train_and_evaluate(texts, labels, n_folds=2)
        assert len(result) == 5

    def test_cv_preds_length_matches_input(self, corpus):
        texts, labels = corpus
        _, _, cv_preds, _, _ = train_and_evaluate(texts, labels, n_folds=2)
        assert len(cv_preds) == len(texts)

    def test_cv_probas_shape(self, corpus):
        texts, labels = corpus
        _, _, _, cv_probas, _ = train_and_evaluate(texts, labels, n_folds=2)
        n_classes = len(set(labels))
        assert cv_probas.shape == (len(texts), n_classes)

    def test_cv_probas_sum_to_one(self, corpus):
        texts, labels = corpus
        _, _, _, cv_probas, _ = train_and_evaluate(texts, labels, n_folds=2)
        row_sums = cv_probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-5)

    def test_metrics_has_required_keys(self, corpus):
        texts, labels = corpus
        _, _, _, _, metrics = train_and_evaluate(texts, labels, n_folds=2)
        required = {"report_text", "report_dict", "confusion_matrix",
                    "labels", "fold_accuracies", "mean_cv_accuracy"}
        assert required.issubset(metrics.keys())

    def test_mean_cv_accuracy_is_float_in_range(self, corpus):
        texts, labels = corpus
        _, _, _, _, metrics = train_and_evaluate(texts, labels, n_folds=2)
        acc = metrics["mean_cv_accuracy"]
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_label_set_in_metrics(self, corpus):
        texts, labels = corpus
        _, _, _, _, metrics = train_and_evaluate(texts, labels, n_folds=2)
        assert set(metrics["labels"]) == set(labels)

    def test_cv_preds_only_known_labels(self, corpus):
        texts, labels = corpus
        _, _, cv_preds, _, _ = train_and_evaluate(texts, labels, n_folds=2)
        assert set(cv_preds).issubset(set(labels))

    def test_fold_accuracies_count(self, corpus):
        texts, labels = corpus
        _, _, _, _, metrics = train_and_evaluate(texts, labels, n_folds=3)
        assert len(metrics["fold_accuracies"]) == 3
