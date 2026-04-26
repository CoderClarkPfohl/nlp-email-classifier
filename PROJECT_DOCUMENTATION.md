# NLP Job Application Email Classifier — Technical Documentation

**Course:** CS 7347 Natural Language Processing
**Team:** Clark Pfohl, Michael Zimmerman, Oleksandra Zolotarevych

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Classification Models](#classification-models)
5. [NLP Components](#nlp-components)
6. [Synthetic Data Generation](#synthetic-data-generation)
7. [Training & Evaluation](#training--evaluation)
8. [Output & Visualizations](#output--visualizations)
9. [Key Results](#key-results)
10. [Design Decisions](#design-decisions)
11. [File Reference](#file-reference)
12. [How to Run](#how-to-run)

---

## 1. Project Overview

This project classifies job application emails into six actionable categories using a multi-stage NLP pipeline. The system ingests a CSV of raw recruiter emails, runs them through preprocessing, rule-based labeling, sentiment analysis, entity extraction, extractive summarization, and finally a machine learning classifier. It produces a color-coded Excel spreadsheet mapping every email to its predicted category, along with charts and metrics.

### The Six Categories

| Category | Description | Example Signal |
|---|---|---|
| `acceptance` | Job offer extended | "pleased to offer you the position" |
| `rejection` | Application declined | "regret to inform you", "not moving forward" |
| `interview` | Interview invitation or scheduling | "schedule an interview", "phone screen" |
| `action_required` | Assessment, coding challenge, form to complete | "complete your assessment", "HackerRank" |
| `in_process` | Application received, under review | "received your application", "under review" |
| `unrelated` | Newsletters, job alerts, portal signups | "unsubscribe", "job alert", "password reset" |

---

## 2. Architecture

```
                        +------------------+
                        |   Raw Email CSV  |
                        +--------+---------+
                                 |
                    [1] Load & validate
                                 |
                    [2] Preprocess text
                         |            |
                   clean_body    model_body
                  (for TF-IDF)  (for DeBERTa)
                         |
                    [3] Rule-based scoring labeler
                         |
                    [4] Sentiment analysis
                         |
                    [5] Entity extraction
                         |
                    [6] Extractive summarization
                         |
              +----------+----------+
              |                     |
         [7a] Ensemble         [7b] DeBERTa
        (SVM+LR+NB)          (zero-shot NLI)
              |                     |
              +----------+----------+
                         |
                    Final labels
                         |
              +----------+----------+
              |          |          |
          Excel CSV   Plots    Metrics
```

The pipeline is sequential: each stage enriches the DataFrame with new columns. The classification step (7) branches based on `--model` flag: `ensemble` (default, no GPU) or `deberta` (requires torch + transformers).

---

## 3. Pipeline Stages

### Stage 1: Load Data (`main.py:load_data`)
- Reads the input CSV via pandas
- Drops rows with empty `email_body` fields
- Expected columns: `sender`, `date`, `email_body`, `company`, `subject`

### Stage 2: Preprocessing (`utils/preprocessing.py`)
Two parallel text representations are created:

**`clean_body`** (for TF-IDF models):
1. Strip HTML tags and entities (`strip_html`)
2. Remove URLs (`remove_urls`)
3. Remove email addresses (`remove_email_addresses`)
4. Remove non-alphanumeric characters except basic punctuation
5. Normalize whitespace
6. Lowercase everything

**`model_body`** (for transformer models):
1. Strip HTML tags
2. Remove URLs
3. Normalize whitespace
4. Truncate to ~2000 characters (rough 512-token proxy)
5. Preserves casing and punctuation (transformers benefit from natural text)

### Stage 3: Rule-Based Labeling (`models/rule_labeler.py`)
A scoring-based heuristic system that generates pseudo-labels for training. This is NOT the final classifier — it provides training signal for the ensemble.

**How it works:**
1. Six phrase dictionaries define signals for each category (e.g., `REJECTION_STRONG`, `INTERVIEW_MEDIUM`)
2. Each category accumulates a score by counting phrase matches, weighted by signal strength:
   - Strong signals: 4-5 points per match
   - Medium signals: 1.5-2 points per match
   - Subject-line bonuses: +2-3 points for key subject phrases
3. The highest-scoring category wins
4. Confidence is based on the margin between the top two scores:
   - Margin >= 5.0 -> 0.95 confidence
   - Margin >= 3.0 -> 0.85
   - Margin >= 1.5 -> 0.70
   - Below 1.5 -> 0.55
5. If no phrases match at all, defaults to `in_process` with 0.30 confidence

**Why scoring instead of if/else:** Real emails often contain signals for multiple categories (e.g., "thank you for applying... please complete the assessment"). The scoring system picks the strongest signal rather than stopping at the first match.

### Stage 4: Sentiment Analysis (`models/sentiment.py`)
A domain-specific lexicon-based sentiment analyzer tuned for recruiter language.

- Three custom word lists: `POSITIVE_WORDS`, `NEGATIVE_WORDS`, `NEUTRAL_WORDS` (all frozensets for O(1) lookup)
- Scans the lowercased text for both single words and multi-word phrases
- Computes:
  - `positive`, `negative`, `neutral` scores (proportional to matches)
  - `compound` score (-1 to +1): `(pos_count - neg_count) / max(pos_count + neg_count, 1)`
  - `label`: "positive" if compound > 0.15, "negative" if < -0.15, else "neutral"

**Domain tuning:** Words like "competitive" and "difficult decision" are negative in this context (they signal rejection), even though they're neutral in general English.

### Stage 5: Entity Extraction (`utils/entity_extraction.py`)
Regex-based extraction of structured information:

| Entity | Method | Example Patterns |
|---|---|---|
| Job role | Regex patterns like `"applied for the X position"` | "Data Analyst", "Software Engineer" |
| Contact person | Looks for names after "sincerely,", "recruiter:", etc. | "Sarah Johnson" |
| Contact email | Finds email addresses, skips noreply/auto addresses | "recruiter@company.com" |
| Dates mentioned | Multiple date format patterns (MM/DD/YYYY, Month DD, YYYY, etc.) | "January 15, 2025" |

### Stage 6: Extractive Summarization (`utils/summarizer.py`)
Selects the most informative sentences from each email.

**Sentence scoring factors:**
- **Keyword relevance:** High-importance words ("offer", "interview", "rejected") score +3.0; medium words ("application", "review") score +1.0
- **Position bonus:** First/second sentences get +2.0; last sentence gets +1.0
- **Length preference:** 8-30 words get +1.0; under 5 words get -2.0
- **Boilerplate penalty:** Sentences containing "unsubscribe", "privacy policy", etc. get -5.0

Top-scored sentences are returned in their original order (preserving narrative flow).

---

## 4. Classification Models

### Ensemble Classifier (`models/svm_classifier.py`)

A soft-voting ensemble of three calibrated classifiers:

| Model | Weight | Why |
|---|---|---|
| Linear SVM (calibrated via sigmoid) | 2.0 | Strong on sparse high-dimensional text; `class_weight="balanced"` handles imbalance |
| Logistic Regression | 2.0 | Well-calibrated probabilities; good generalization |
| Multinomial Naive Bayes | 1.0 | Fast; handles sparse features; good with class priors |

**TF-IDF Features:**
- Max 8,000 features
- Unigrams + bigrams (`ngram_range=(1, 2)`)
- Sublinear TF scaling (`sublinear_tf=True`)
- Min document frequency 2, max document frequency 95%

**Oversampling:**
Minority classes (< 30 samples in training fold) are duplicated to reach 30 samples via random sampling with replacement. This happens inside each CV fold to prevent data leakage.

**Evaluation:** 5-fold stratified cross-validation. Each fold trains on oversampled training data, predicts on the held-out test fold. Final predictions are the concatenation of all test-fold predictions.

### DeBERTa Zero-Shot Classifier (`models/deberta_classifier.py`)

Uses Hugging Face's `zero-shot-classification` pipeline with NLI (Natural Language Inference) models. No training data required.

**Key design choices:**

1. **Contrastive candidate labels:** Instead of short labels like "rejection", uses full descriptive sentences that force the NLI model to discriminate on the action:
   - "the company is rejecting or declining the candidate's application"
   - "the company has received the application and is still reviewing it"

2. **Hypothesis template:** Wraps each label in `"This email is about {}."` so the model reasons about meaning, not keyword overlap.

3. **Model presets:** Four tiers selectable via `--deberta-model`:
   - `best`: deberta-v3-large-zeroshot-v2 (~60 min CPU, ~93-96%)
   - `balanced`: deberta-v3-base-zeroshot-v1 (~20 min CPU, ~91-93%) [default]
   - `fast`: nli-deberta-v3-small (~12 min CPU, ~87-89%)
   - `bart`: bart-large-mnli (~45 min CPU, ~90-92%)

4. **Hybrid tie-breaker:** When DeBERTa confidence < 0.65 AND it disagrees with the rule label, falls back to the rule label. This catches cases where DeBERTa is uncertain.

---

## 5. Synthetic Data Generation (`generate_synthetic_data.py`)

Generates 2,000 class-balanced emails from randomized templates to augment the small real dataset.

**Generation method:**
- 8-10 templates per category, each with randomized:
  - Company names (120+ real companies: FAANG, finance, healthcare, etc.)
  - Job roles (45+ titles across data, engineering, product, finance)
  - Recruiter names (23 names)
  - Dates (random within Jan-May 2025)
  - Job IDs, salaries, interview formats
- Target distribution: 300 acceptance, 400 rejection, 350 interview, 350 action_required, 400 in_process, 200 unrelated

**Why synthetic data helps less for TF-IDF:**
TF-IDF + SVM is vocabulary-dependent. Synthetic templates use clean, predictable phrasing ("we regret to inform you") while real recruiter emails use varied language with different vocabulary. The TF-IDF features don't transfer well between domains. This is actually a strong argument for transformer models like DeBERTa, which understand semantics rather than matching exact words.

---

## 6. Training & Evaluation (`train_with_synthetic.py`)

Runs three experiments and compares:

1. **Real data only** — Ensemble with 5-fold CV on 494 real emails
2. **Synthetic only -> Real** — Train on 2,000 synthetic, test on all 494 real
3. **Combined** — All synthetic + train-fold of real -> test on real (5-fold CV)

Each experiment uses the same TF-IDF + ensemble architecture with oversampling. Produces side-by-side confusion matrices and an accuracy comparison bar chart.

---

## 7. Output & Visualizations

### Output Files (in `results/`)

| File | Description |
|---|---|
| `classified_emails.xlsx` | Color-coded Excel spreadsheet with every email and its predicted category |
| `classified_emails.csv` | Same data as flat CSV (no formatting) |
| `classification_report.txt` | Precision, recall, F1 per class + overall accuracy |
| `accuracy_analysis.txt` | Written analysis of accuracy bottlenecks and ceilings |
| `confusion_matrix.png` | Heatmap of predicted vs. true labels |
| `accuracy_comparison.png` | Bar chart comparing all approaches |
| `label_distribution.png` | Category distribution + mean sentiment by category |
| `sentiment_distribution.png` | Sentiment score histogram + pie chart |
| `top_companies.png` | Stacked bar chart of categories per top 15 companies |
| `confusion_matrices_comparison.png` | Synthetic vs. combined side-by-side |

### Color Coding (Excel output)

Each category has a distinct color in the exported Excel file:
- **acceptance** — Green
- **rejection** — Red
- **interview** — Blue
- **action_required** — Orange
- **in_process** — Yellow
- **unrelated** — Gray

---

## 8. Key Results

| Approach | Accuracy on Real Data |
|---|---|
| Majority baseline (always "in_process") | 64.4% |
| Real data only — ensemble + oversampling | **88.1%** |
| Synthetic only -> real | 72.5% |
| Combined synthetic + real | 86.4% |
| DeBERTa zero-shot (estimated) | ~94% |
| Fine-tuned DeBERTa (estimated) | ~97% |

### Per-Class Error Rates
- **interview** (8 emails): 100% error rate — too few training examples
- **action_required** (12 emails): 67% error rate — confused with in_process
- **rejection** (96 emails): 21% error rate — polite rejections read like confirmations
- **unrelated** (60 emails): 13% error rate — job alerts overlap with confirmations
- **in_process** (318 emails): 5% error rate — dominant class, well-learned

---

## 9. Design Decisions

**Ensemble over single SVM:** Soft voting across three classifiers smooths out individual model weaknesses. SVM handles high-dimensional sparse text; LR gives calibrated probabilities; NB handles class priors well.

**Duplication over SMOTE:** SMOTE creates synthetic feature vectors by interpolating in TF-IDF space, which produces nonsensical combinations for text. Simple duplication preserves real email patterns.

**Scoring-based rule labeler over if/else:** Emails often contain signals for multiple categories. The scoring system accumulates evidence and picks the strongest, with confidence based on margin. More robust than first-match if/else chains.

**Two text representations:** TF-IDF models work best on heavily cleaned, lowercased text. Transformers (DeBERTa) work best on natural text with casing and punctuation preserved. Both are computed in Stage 2.

**DeBERTa as zero-shot:** The class imbalance (8 interview, 0 acceptance in real data) makes supervised learning impossible for rare classes. Zero-shot classification bypasses this entirely since it doesn't need training examples.

---

## 10. File Reference

```
nlp_project/
├── main.py                        # Main 7-stage pipeline entry point
├── generate_synthetic_data.py     # Creates 2000 balanced synthetic emails
├── train_with_synthetic.py        # Three-experiment comparison script
├── requirements.txt               # Dependencies (pandas, sklearn, matplotlib, openpyxl)
├── README.md                      # Quick-start README
├── CLAUDE.md                      # Session notes
├── PROJECT_DOCUMENTATION.md       # This file
├── data/
│   ├── job_app_confirmation_emails_anonymized.csv   # 494 real emails
│   └── synthetic_emails.csv                          # 2000 synthetic emails
├── models/
│   ├── __init__.py
│   ├── rule_labeler.py            # v2 scoring-based heuristic labeler
│   ├── svm_classifier.py          # Ensemble (SVM+LR+NB) with oversampling
│   ├── deberta_classifier.py      # Zero-shot DeBERTa classifier
│   └── sentiment.py               # Domain-specific sentiment lexicon
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py           # Text cleaning and tokenization
│   ├── entity_extraction.py       # Company, role, contact, date extraction
│   ├── summarizer.py              # Extractive email summarizer
│   └── excel_export.py            # Color-coded Excel export
├── tests/
│   ├── test_excel_export.py       # Tests for Excel export
│   ├── test_preprocessing.py      # Tests for text preprocessing
│   ├── test_rule_labeler.py       # Tests for rule-based labeling
│   ├── test_sentiment.py          # Tests for sentiment analysis
│   ├── test_summarizer.py         # Tests for summarization
│   └── test_entity_extraction.py  # Tests for entity extraction
└── results/
    ├── classified_emails.xlsx     # Color-coded Excel output
    ├── classified_emails.csv      # Flat CSV output
    ├── classification_report.txt  # Metrics report
    └── *.png                      # Various charts
```

---

## 11. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Basic pipeline (ensemble classifier, no GPU needed)
python main.py --input data/job_app_confirmation_emails_anonymized.csv

# DeBERTa zero-shot (requires torch + transformers)
pip install torch "transformers>=4.30,<5.0" sentencepiece protobuf
python main.py --input data/job_app_confirmation_emails_anonymized.csv --model deberta

# Generate synthetic training data
python generate_synthetic_data.py

# Train with synthetic + real combined, see comparison
python train_with_synthetic.py

# Run tests
python -m pytest tests/ -v
```
