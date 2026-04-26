"""
Per-category n-gram language models for email classification.

Demonstrates Language Modeling (CS 7347 Topic 3: N-grams and Markov assumption).

Builds a smoothed bigram language model for each email category,
then computes per-email perplexity under each model. Lower perplexity
under a category's LM means the email's word transition patterns match
that category's typical language — e.g., rejection emails frequently
contain the bigram "regret to" while acceptance emails contain "pleased to".

Theory
------
- A bigram LM estimates P(w_i | w_{i-1}) from training data
- Laplace (add-k) smoothing assigns nonzero probability to unseen bigrams
- Perplexity PP(W) = exp(-1/N * sum log P(w_i | w_{i-1})) measures how well
  the model "predicts" the text — lower = better fit
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


class NgramLanguageModel:
    """
    Smoothed n-gram language model with Laplace (add-k) smoothing.

    Implements the Markov assumption: P(w_i | w_{i-1}, ..., w_{i-n+1})
    estimated from n-gram counts, with add-k smoothing to handle
    unseen n-grams (avoids zero-probability problem).
    """

    def __init__(self, n: int = 2, k: float = 0.1):
        """
        Parameters
        ----------
        n : int
            Order of the language model (2 = bigram, 3 = trigram).
        k : float
            Laplace smoothing parameter. k=1 is full add-one smoothing;
            k<1 gives less aggressive smoothing (better for sparse data).
        """
        self.n = n
        self.k = k
        self.ngram_counts: Dict[tuple, Counter] = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.vocab: set = set()

    def fit(self, tokenized_texts: List[List[str]]) -> "NgramLanguageModel":
        """Train on a list of tokenized documents."""
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()

        for tokens in tokenized_texts:
            # Pad with start/end symbols
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for token in padded:
                self.vocab.add(token)
            # Count n-grams
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1 : i])
                word = padded[i]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

        return self

    def log_prob_sequence(self, tokens: List[str]) -> Tuple[float, int]:
        """
        Compute total log probability of a token sequence.

        Returns (log_probability, token_count) where token_count
        includes the </s> end symbol.
        """
        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        V = max(len(self.vocab), 1)
        log_prob = 0.0
        count = 0

        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1 : i])
            word = padded[i]
            # Smoothed conditional probability: P(word | context)
            numerator = self.ngram_counts[context][word] + self.k
            denominator = self.context_counts[context] + self.k * V
            log_prob += math.log(numerator / denominator)
            count += 1

        return log_prob, count

    def perplexity(self, tokens: List[str]) -> float:
        """
        Compute perplexity of a token sequence.

        PP(W) = exp(-1/N * log P(W))

        Lower perplexity = the model assigns higher probability to this
        text = better fit between text and model's training distribution.
        """
        if not tokens:
            return float("inf")
        log_prob, count = self.log_prob_sequence(tokens)
        if count == 0:
            return float("inf")
        return math.exp(-log_prob / count)


class CategoryLanguageModels:
    """
    Collection of per-category bigram language models.

    Trains one LM per email category, then uses perplexity under each
    model as a classification feature. The intuition: if an email has
    low perplexity under the "rejection" LM but high perplexity under
    the "acceptance" LM, it likely contains rejection-like word patterns.

    This is analogous to the Naive Bayes generative approach but at the
    bigram level rather than unigram (bag of words).
    """

    def __init__(self, n: int = 2, k: float = 0.1):
        self.n = n
        self.k = k
        self.models: Dict[str, NgramLanguageModel] = {}
        self.categories: List[str] = []

    def fit(
        self, tokenized_texts: List[List[str]], labels: List[str]
    ) -> "CategoryLanguageModels":
        """Train one language model per unique label."""
        self.categories = sorted(set(labels))
        self.models = {}

        for cat in self.categories:
            cat_texts = [t for t, l in zip(tokenized_texts, labels) if l == cat]
            lm = NgramLanguageModel(n=self.n, k=self.k)
            lm.fit(cat_texts)
            self.models[cat] = lm

        return self

    def perplexity_features(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        Compute perplexity of each text under each category's LM.

        Returns
        -------
        ndarray of shape (n_texts, n_categories)
            Log-transformed perplexity values. Lower values in column j
            indicate the text fits category j's language model well.
            Log-transform (log1p) compresses the scale for ML classifiers.
        """
        n = len(tokenized_texts)
        features = np.zeros((n, len(self.categories)))

        for j, cat in enumerate(self.categories):
            lm = self.models[cat]
            for i, tokens in enumerate(tokenized_texts):
                features[i, j] = lm.perplexity(tokens)

        # Log-transform to compress scale (perplexity can span orders of magnitude)
        return np.log1p(features)
