"""
Enhanced feature engineering for email classification.

Combines multiple NLP feature types alongside TF-IDF to create
a richer representation of emails for the ensemble classifier.

Demonstrates concepts from multiple CS 7347 topics:
  - Topic 2 (Information Retrieval): cosine similarity in vector space
  - Topic 3 (Language Modeling): n-gram perplexity features (via ngram_lm)
  - Topic 4 (Vector Semantics): document similarity via TF-IDF centroids
  - Topic 5 (Supervised ML): domain-specific feature extraction
  - Topic 6 (Sequence Labeling / NER): entity counts as features
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple

from utils.preprocessing import tokenize
from models.ngram_lm import CategoryLanguageModels
from models.rule_labeler import (
    REJECTION_STRONG,
    REJECTION_MEDIUM,
    ACCEPTANCE_PHRASES,
    INTERVIEW_STRONG,
    INTERVIEW_MEDIUM,
    ACTION_STRONG,
    ACTION_MEDIUM,
    UNRELATED_PHRASES,
    IN_PROCESS_PHRASES,
)


# ─────────────────────────────────────────────────────────────
#  Keyword / domain features (Topic 5: feature extraction)
# ─────────────────────────────────────────────────────────────

# Map each category to its phrase lists from the rule labeler
_CATEGORY_PHRASE_LISTS = {
    "acceptance": [ACCEPTANCE_PHRASES],
    "rejection": [REJECTION_STRONG, REJECTION_MEDIUM],
    "interview": [INTERVIEW_STRONG, INTERVIEW_MEDIUM],
    "action_required": [ACTION_STRONG, ACTION_MEDIUM],
    "in_process": [IN_PROCESS_PHRASES],
    "unrelated": [UNRELATED_PHRASES],
}

KEYWORD_CATEGORIES = sorted(_CATEGORY_PHRASE_LISTS.keys())


def compute_keyword_features(texts: List[str]) -> np.ndarray:
    """
    Count phrase matches from the rule labeler's domain dictionaries.

    Each column represents a category; the value is the total number
    of matching phrases from that category's phrase lists found in
    the text. This gives the ML model explicit access to the domain
    knowledge encoded in the rule labeler.

    Returns
    -------
    ndarray of shape (n_texts, 6)
        One column per category (sorted alphabetically).
    """
    features = np.zeros((len(texts), len(KEYWORD_CATEGORIES)), dtype=np.float64)

    for i, text in enumerate(texts):
        lower = text.lower()
        for j, cat in enumerate(KEYWORD_CATEGORIES):
            count = 0
            for phrase_list in _CATEGORY_PHRASE_LISTS[cat]:
                count += sum(1 for p in phrase_list if p in lower)
            features[i, j] = count

    return features


# ─────────────────────────────────────────────────────────────
#  NER count features (Topic 6: Named Entity Recognition)
# ─────────────────────────────────────────────────────────────


def compute_ner_features(
    extracted_roles: List,
    contact_persons: List,
    contact_emails: List,
    dates_mentioned: List,
) -> np.ndarray:
    """
    Convert entity extraction results into numeric features.

    NER-derived features capture structural patterns: acceptance emails
    mention contact persons and start dates; action_required emails
    mention multiple deadlines; unrelated emails rarely mention roles.

    Features (4 columns):
        has_job_role, has_contact_person, has_contact_email, n_dates
    """
    n = len(extracted_roles)
    features = np.zeros((n, 4), dtype=np.float64)

    for i in range(n):
        features[i, 0] = 1.0 if extracted_roles[i] else 0.0
        features[i, 1] = 1.0 if contact_persons[i] else 0.0
        features[i, 2] = 1.0 if contact_emails[i] else 0.0
        # Count dates
        dates = dates_mentioned[i]
        if isinstance(dates, str) and dates.strip():
            features[i, 3] = float(len([d for d in dates.split(";") if d.strip()]))
        elif isinstance(dates, list):
            features[i, 3] = float(len(dates))

    return features


# ─────────────────────────────────────────────────────────────
#  Cosine similarity to category centroids (Topics 2 & 4)
# ─────────────────────────────────────────────────────────────


def compute_category_centroids(
    X_tfidf,
    labels: List[str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Compute the TF-IDF centroid (mean vector) for each category.

    The centroid represents the category's "position" in the vector
    space model (Topic 2). Documents near a centroid share vocabulary
    distribution with that category.
    """
    categories = sorted(set(labels))
    centroids = {}

    for cat in categories:
        mask = [i for i, l in enumerate(labels) if l == cat]
        if mask:
            cat_vectors = X_tfidf[mask]
            centroid = np.asarray(cat_vectors.mean(axis=0)).flatten()
            centroids[cat] = centroid

    return centroids, categories


def compute_cosine_features(
    X_tfidf,
    centroids: Dict[str, np.ndarray],
    categories: List[str],
) -> np.ndarray:
    """
    Compute cosine similarity between each document and each
    category centroid.

    Cosine similarity measures the angle between vectors in the
    TF-IDF space (Topic 2: vector space model). High similarity
    to a category's centroid indicates shared vocabulary patterns
    (Topic 4: distributional semantics — similar contexts imply
    similar meaning).

    Returns
    -------
    ndarray of shape (n_docs, n_categories)
        Cosine similarity in [0, 1] (TF-IDF vectors are non-negative).
    """
    n = X_tfidf.shape[0]
    features = np.zeros((n, len(categories)), dtype=np.float64)

    for j, cat in enumerate(categories):
        centroid = centroids[cat].reshape(1, -1)
        sims = cosine_similarity(X_tfidf, centroid).flatten()
        features[:, j] = sims

    return features


# ─────────────────────────────────────────────────────────────
#  Sentiment features
# ─────────────────────────────────────────────────────────────


def compute_sentiment_features(
    sentiment_compounds: List,
    sentiment_labels: List,
) -> np.ndarray:
    """
    Encode sentiment analysis output as numeric features.

    Features (4 columns):
        compound_shifted (0-1 range), is_positive, is_negative, is_neutral

    Note: compound is shifted from [-1,1] to [0,1] to ensure non-negative
    values for MultinomialNB compatibility.
    """
    n = len(sentiment_compounds)
    features = np.zeros((n, 4), dtype=np.float64)

    for i in range(n):
        compound = sentiment_compounds[i]
        features[i, 0] = (float(compound) + 1.0) / 2.0 if compound else 0.5
        label = str(sentiment_labels[i]) if sentiment_labels[i] else "neutral"
        if label == "positive":
            features[i, 1] = 1.0
        elif label == "negative":
            features[i, 2] = 1.0
        else:
            features[i, 3] = 1.0

    return features


# ─────────────────────────────────────────────────────────────
#  Leakage-free enhanced feature transformer
# ─────────────────────────────────────────────────────────────


class EnhancedFeatureTransformer:
    """
    Fit/transform wrapper for enhanced features that learn from labels.

    Keyword, NER, and sentiment features are row-local and can be computed
    directly. Language-model and centroid features must be fit only on the
    training split, otherwise cross-validation sees information from held-out
    rows and reports optimistic metrics.
    """

    def __init__(
        self,
        include_lm: bool = True,
        include_centroids: bool = True,
        include_keywords: bool = True,
        include_ner: bool = True,
        include_sentiment: bool = True,
        text_column: str = "clean_body",
    ):
        self.include_lm = include_lm
        self.include_centroids = include_centroids
        self.include_keywords = include_keywords
        self.include_ner = include_ner
        self.include_sentiment = include_sentiment
        self.text_column = text_column

        self.lm_ = None
        self.centroid_tfidf_ = None
        self.centroids_ = {}
        self.categories_ = []
        self.n_features_out_ = None
        self.is_fitted_ = False

    def fit(self, df_train, labels_train: List[str]) -> "EnhancedFeatureTransformer":
        """Fit train-dependent enhanced feature components."""
        texts = self._texts(df_train)
        labels = list(labels_train)
        self.categories_ = sorted(set(labels))

        if self.include_lm:
            tokenized = [tokenize(t) for t in texts]
            self.lm_ = CategoryLanguageModels(n=2, k=0.1)
            self.lm_.fit(tokenized, labels)

        if self.include_centroids:
            self.centroid_tfidf_ = TfidfVectorizer(
                max_features=5000,
                sublinear_tf=True,
                min_df=1,
            )
            X_train = self.centroid_tfidf_.fit_transform(texts)
            self.centroids_, centroid_categories = compute_category_centroids(
                X_train, labels
            )
            self.categories_ = centroid_categories

        self.is_fitted_ = True
        sample = self.transform(df_train.iloc[:1] if hasattr(df_train, "iloc") else df_train[:1])
        self.n_features_out_ = sample.shape[1]
        return self

    def transform(self, df) -> np.ndarray:
        """Transform rows into enhanced dense features without refitting."""
        if not self.is_fitted_:
            raise ValueError("EnhancedFeatureTransformer must be fit before transform")

        texts = self._texts(df)
        parts = []

        if self.include_lm:
            tokenized = [tokenize(t) for t in texts]
            lm_feats = self.lm_.perplexity_features(tokenized)
            lm_feats = np.nan_to_num(lm_feats, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(lm_feats)

        if self.include_centroids:
            X = self.centroid_tfidf_.transform(texts)
            parts.append(compute_cosine_features(X, self.centroids_, self.categories_))

        if self.include_keywords:
            parts.append(compute_keyword_features(texts))

        if self.include_ner and self._has_columns(
            df, ["extracted_role", "contact_person", "contact_email", "dates_mentioned"]
        ):
            parts.append(
                compute_ner_features(
                    df["extracted_role"].tolist(),
                    df["contact_person"].tolist(),
                    df["contact_email"].tolist(),
                    df["dates_mentioned"].tolist(),
                )
            )

        if self.include_sentiment and self._has_columns(
            df, ["sentiment_compound", "sentiment_label"]
        ):
            parts.append(
                compute_sentiment_features(
                    df["sentiment_compound"].tolist(),
                    df["sentiment_label"].tolist(),
                )
            )

        if not parts:
            return np.zeros((len(texts), 0), dtype=np.float64)
        return np.hstack(parts)

    def fit_transform(self, df_train, labels_train: List[str]) -> np.ndarray:
        """Fit on training rows and return their features."""
        return self.fit(df_train, labels_train).transform(df_train)

    def _texts(self, df) -> List[str]:
        if self.text_column not in df.columns:
            raise ValueError(f"DataFrame is missing required column: {self.text_column}")
        return df[self.text_column].fillna("").astype(str).tolist()

    @staticmethod
    def _has_columns(df, columns: List[str]) -> bool:
        return all(c in df.columns for c in columns)


# ─────────────────────────────────────────────────────────────
#  Combined enhanced features
# ─────────────────────────────────────────────────────────────


def compute_enhanced_features(df, label_column: str = "rule_label") -> np.ndarray:
    """
    Build all enhanced features from an enriched DataFrame.

    Combines:
      1. N-gram LM perplexity features (Topic 3) — n_categories columns
      2. Cosine similarity to TF-IDF centroids (Topics 2 & 4) — n_categories columns
      3. Domain keyword phrase counts (Topic 5) — 6 columns
      4. NER entity count features (Topic 6) — 4 columns (if available)
      5. Sentiment features — 4 columns (if available)

    Returns
    -------
    ndarray of shape (n_emails, n_features)
        Dense feature matrix ready to stack with TF-IDF in the ensemble.
        All values are non-negative (required by MultinomialNB).

    Note: LM and cosine features are computed on the full dataset.
    In a production system these should be computed per-fold to avoid
    minor data leakage, but the impact is negligible for this dataset
    size and the code clarity benefit is significant.
    """
    texts = df["clean_body"].tolist()
    labels = df[label_column].tolist()

    # ── Topic 3: Language Modeling — per-category bigram LMs ──
    tokenized = [tokenize(t) for t in texts]
    lm = CategoryLanguageModels(n=2, k=0.1)
    lm.fit(tokenized, labels)
    lm_feats = lm.perplexity_features(tokenized)

    # ── Topics 2 & 4: Vector Space — cosine similarity to centroids ──
    temp_tfidf = TfidfVectorizer(
        max_features=5000,
        sublinear_tf=True,
        min_df=1,
    )
    X_temp = temp_tfidf.fit_transform(texts)
    centroids, cats = compute_category_centroids(X_temp, labels)
    cosine_feats = compute_cosine_features(X_temp, centroids, cats)

    # ── Topic 5: Domain keyword features ──
    kw_feats = compute_keyword_features(texts)

    parts = [lm_feats, cosine_feats, kw_feats]

    # ── Topic 6: NER entity features ──
    if "extracted_role" in df.columns:
        ner_feats = compute_ner_features(
            df["extracted_role"].tolist(),
            df["contact_person"].tolist(),
            df["contact_email"].tolist(),
            df["dates_mentioned"].tolist(),
        )
        parts.append(ner_feats)

    # ── Sentiment features ──
    if "sentiment_compound" in df.columns:
        sent_feats = compute_sentiment_features(
            df["sentiment_compound"].tolist(),
            df["sentiment_label"].tolist(),
        )
        parts.append(sent_feats)

    return np.hstack(parts)
