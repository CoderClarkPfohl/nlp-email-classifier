#CS 7347 Natural Language Processing


#***Ensemble classifier***

#________________libraries

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
#________________


#oversampler
def oversample_minority(X, y, min_samples: int = 30, random_state: int = 42):
   #duplicate until reach min_samples per class
    rng = np.random.RandomState(random_state)
    counts = Counter(y)
    indices = list(range(len(y)))

    for label, count in counts.items():
        if count < min_samples:
            class_idx = [i for i, lbl in enumerate(y) if lbl == label]
            n_needed = min_samples - count
            extra_idx = rng.choice(class_idx, size=n_needed, replace=True)
            indices.extend(extra_idx.tolist())

    rng.shuffle(indices)

    #matrix
    if hasattr(X, 'toarray'):
        X_new = X[indices]
    else:
        X_new = X[indices]

    y_new = [y[i] for i in indices]
    return X_new, y_new


#softvote
class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    #predicted probabilities averaged across classifiers
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


#builders pipelinne
def build_tfidf(max_features: int = 8000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
    )


def build_ensemble():
    #ensemble (3m) calibrated SVM
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
        weights=[2.0, 2.0, 1.0],  #weights SVM + LR
    )


def train_and_evaluate(
    texts: List[str],
    labels: List[str],
    n_folds: int = 5,
) -> Tuple[object, np.ndarray, Dict]:
    #training + cross-val, returns trained tfidf, ensemble, predictions, metrics
    #out -> (tfidf, ensemble, predictions, metrics_dict)
    tfidf = build_tfidf()

    #text -> TF-IDF matrix
    X_all = tfidf.fit_transform(texts)
    y_all = labels

    label_set = sorted(set(labels))
    print(f"       Training with {len(texts)} samples, {len(label_set)} classes")
    print(f"       Class distribution: {Counter(labels)}")

    #folds 
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_preds = np.array([""] * len(labels), dtype=object)
    cv_probas = np.zeros((len(labels), len(label_set)))

    fold_reports = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        X_train = X_all[train_idx]
        y_train = [y_all[i] for i in train_idx]
        X_test = X_all[test_idx]

        #oversampl
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

    #final evaluation on all data
    report_str = classification_report(y_all, cv_preds, labels=label_set, zero_division=0)
    report_dict = classification_report(y_all, cv_preds, labels=label_set, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_all, cv_preds, labels=label_set)

    #training fin model on all data with oversampling
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

    return tfidf, final_ensemble, cv_preds, metrics


def predict(tfidf, ensemble, texts: List[str]) -> np.ndarray:
    #inference on new texts
    X = tfidf.transform(texts)
    return ensemble.predict(X)
