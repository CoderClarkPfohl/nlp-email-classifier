# Job Application Email Classifier — NLP Project (CS 7347)

## Authors
- Clark Pfohl
- Michael Zimmerman
- Oleksandra Zolotarevych

## Overview
This project reads job application confirmation emails and classifies them into
actionable categories using NLP techniques. Two classification pipelines are
provided:

| Pipeline | Model | Requires GPU? |
|----------|-------|---------------|
| **Baseline (SVM)** | TF-IDF → Linear SVM | No |
| **Transformer (DeBERTa)** | Zero-shot `microsoft/deberta-v3-large` | Recommended |

### Classification Labels
| Label | Meaning |
|-------|---------|
| `acceptance` | Offer letter / congratulations |
| `rejection` | Application declined |
| `interview` | Interview invitation or scheduling |
| `action_required` | Assessment, test, or task to complete |
| `in_process` | Application received / under review |
| `unrelated` | Newsletter, marketing, account setup |

## Quick Start

```bash
# 1. Install dependencies (only needed for DeBERTa pipeline)
pip install torch transformers

# 2. Run full pipeline (works without torch — uses SVM baseline)
python main.py --input data/job_app_confirmation_emails_anonymized.csv

# 3. Run with DeBERTa (requires transformers + torch)
python main.py --input data/job_app_confirmation_emails_anonymized.csv --model deberta
```

## Project Structure
```
nlp_project/
├── main.py                  # Entry point — runs full pipeline
├── README.md
├── data/                    # Input CSV lives here
├── models/
│   ├── rule_labeler.py      # Heuristic labeler (generates training labels)
│   ├── svm_classifier.py    # TF-IDF + SVM pipeline
│   ├── deberta_classifier.py# Zero-shot DeBERTa classifier
│   └── sentiment.py         # Lexicon-based sentiment analysis
├── utils/
│   ├── preprocessing.py     # Text cleaning & tokenization
│   ├── entity_extraction.py # Company / contact / date detection
│   └── summarizer.py        # Extractive email summarizer
├── results/                 # Output CSVs, plots, metrics
└── requirements.txt
```

## Pipeline Diagram
```
Raw Emails (CSV)
       │
       ▼
  Preprocessing (clean HTML, normalize, tokenize)
       │
       ├──► Sentiment Analysis (lexicon scores)
       ├──► Entity Extraction (company, role, contact, dates)
       ├──► Extractive Summarization
       │
       ▼
  Classification
       ├── Rule-based labeler (heuristic pseudo-labels)
       ├── TF-IDF + SVM (trained on pseudo-labels)
       └── DeBERTa zero-shot (no training required)
       │
       ▼
  Evaluation & Results
       ├── Classification report (precision, recall, F1)
       ├── Confusion matrix
       ├── Sentiment distribution
       └── Final labeled CSV
```
