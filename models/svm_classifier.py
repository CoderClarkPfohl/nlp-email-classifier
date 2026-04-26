"""
Ensemble classifier for job application emails.
Combines TF-IDF features with multiple classifiers and handles
severe class imbalance through synthetic oversampling.

Models in the ensemble:
  1. Linear SVM (strong on text, handles high-dim features)
  2. Logistic Regression (probabilistic, good calibration)
  3. Multinomial Naive Bayes (fast, handles sparse features well)

The ensemble uses soft voting (averaged probabilities) for the final
prediction, which is more robust than any single model.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, List, Tuple
from collections import Counter


# ─────────────────────────────────────────────────────────────
#  Simple oversampler (no imblearn dependency)
# ─────────────────────────────────────────────────────────────
def oversample_minority(X, y, min_samples: int = 30, random_state: int = 42):
    """
    Duplicate samples from minority classes until they reach min_samples.
    This is a simple alternative to SMOTE that works with sparse TF-IDF matrices.
    """
    rng = np.random.RandomState(random_state)
    counts = Counter(y)
    indices = list(range(len(y)))

    for label, count in counts.items():
        if count < min_samples:
            # Find indices of this class
            class_idx = [i for i, lbl in enumerate(y) if lbl == label]
            # How many more do we need?
            n_needed = min_samples - count
            # Sample with replacement
            extra_idx = rng.choice(class_idx, size=n_needed, replace=True)
            indices.extend(extra_idx.tolist())

    rng.shuffle(indices)

    if hasattr(X, 'toarray'):
        # Sparse matrix
        X_new = X[indices]
    else:
        X_new = X[indices]

    y_new = [y[i] for i in indices]
    return X_new, y_new


# ─────────────────────────────────────────────────────────────
#  Soft-voting ensemble
# ─────────────────────────────────────────────────────────────
class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Soft-voting ensemble that averages predicted probabilities from
    multiple calibrated classifiers.
    """
    def __init__(self, classifiers=None, weights=None):
        self.classifiers = classifiers or []
        self.weights = weights
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        for clf in self.classifiers:
            clf.fit(X, y)
        return self

    def predict_proba(self, X):
        probas = []
        for clf in self.classifiers:
            probas.append(clf.predict_proba(X))

        if self.weights:
            weighted = [p * w for p, w in zip(probas, self.weights)]
            avg = np.sum(weighted, axis=0) / sum(self.weights)
        else:
            avg = np.mean(probas, axis=0)
        return avg

    def predict(self, X):
        avg_proba = self.predict_proba(X)
        return self.classes_[np.argmax(avg_proba, axis=1)]


# ─────────────────────────────────────────────────────────────
#  Pipeline builders
# ─────────────────────────────────────────────────────────────
def build_tfidf(max_features: int = 8000, use_stemming: bool = False) -> TfidfVectorizer:
    kwargs = dict(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
    )
    if use_stemming:
        from utils.preprocessing import stemming_tokenizer

        kwargs["tokenizer"] = stemming_tokenizer
        kwargs["token_pattern"] = None  # suppress sklearn warning
    return TfidfVectorizer(**kwargs)


def build_ensemble():
    """Build the 3-model ensemble with calibrated probabilities."""
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
        cv=3, method="sigmoid"
    )
    lr = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000,
        solver="lbfgs", random_state=42
    )
    nb = MultinomialNB(alpha=0.5)

    return SoftVotingEnsemble(
        classifiers=[svm, lr, nb],
        weights=[2.0, 2.0, 1.0],  # SVM and LR weighted higher
    )


def train_and_evaluate(
    texts: List[str],
    labels: List[str],
    n_folds: int = 5,
    extra_features: np.ndarray = None,
    use_stemming: bool = False,
) -> Tuple[object, object, np.ndarray, np.ndarray, Dict]:
    """
    Train the ensemble pipeline with oversampling and evaluate via
    stratified k-fold cross-validation.

    Parameters
    ----------
    extra_features : ndarray, optional
        Dense feature matrix to stack alongside TF-IDF (e.g., LM perplexity,
        cosine similarity, keyword counts, NER features, sentiment).
    use_stemming : bool
        If True, apply suffix-stripping stemmer during TF-IDF tokenization.

    Returns
    -------
    (tfidf, ensemble, cv_preds, cv_probas, metrics_dict)

    cv_probas : ndarray of shape (n_samples, n_classes)
        Per-class probabilities for each sample from its held-out fold.
    """
    tfidf = build_tfidf(use_stemming=use_stemming)

    # Transform all text to TF-IDF
    X_tfidf = tfidf.fit_transform(texts)

    # Stack extra features alongside TF-IDF if provided
    if extra_features is not None:
        from scipy.sparse import hstack as sp_hstack, csr_matrix

        X_all = sp_hstack([X_tfidf, csr_matrix(extra_features)], format="csr")
    else:
        X_all = X_tfidf
    y_all = labels

    label_set = sorted(set(labels))
    print(f"       Training with {len(texts)} samples, {len(label_set)} classes")
    print(f"       Class distribution: {Counter(labels)}")

    # ── Cross-validation with oversampling inside each fold ──
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_preds = np.array([""] * len(labels), dtype=object)
    cv_probas = np.zeros((len(labels), len(label_set)))

    fold_reports = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        X_train = X_all[train_idx]
        y_train = [y_all[i] for i in train_idx]
        X_test = X_all[test_idx]

        # Oversample minority classes in training set only
        X_train_os, y_train_os = oversample_minority(X_train, y_train, min_samples=25)

        ensemble = build_ensemble()
        ensemble.fit(X_train_os, y_train_os)

        preds = ensemble.predict(X_test)
        probas = ensemble.predict_proba(X_test)

        cv_preds[test_idx] = preds
        cv_probas[test_idx] = probas

        fold_acc = np.mean(preds == np.array([y_all[i] for i in test_idx]))
        fold_reports.append(fold_acc)
        print(f"       Fold {fold_idx+1}: accuracy = {fold_acc:.4f}")

    # ── Final metrics ──
    report_str = classification_report(y_all, cv_preds, labels=label_set,
                                        zero_division=0)
    report_dict = classification_report(y_all, cv_preds, labels=label_set,
                                         output_dict=True, zero_division=0)
    cm = confusion_matrix(y_all, cv_preds, labels=label_set)

    # ── Train final model on all data (for inference) ──
    X_all_os, y_all_os = oversample_minority(X_all, list(y_all), min_samples=25)
    final_ensemble = build_ensemble()
    final_ensemble.fit(X_all_os, y_all_os)

    metrics = {
        "report_text": report_str,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "labels": label_set,
        "fold_accuracies": fold_reports,
        "mean_cv_accuracy": np.mean(fold_reports),
    }

    return tfidf, final_ensemble, cv_preds, cv_probas, metrics


def predict(tfidf, ensemble, texts: List[str]) -> np.ndarray:
    """Run inference on new texts."""
    X = tfidf.transform(texts)
    return ensemble.predict(X)
