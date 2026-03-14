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

### Ensemble (works on any OS, no GPU needed)

```bash
pip install -r requirements.txt
python main.py --input data/job_app_confirmation_emails_anonymized.csv
```

### DeBERTa — macOS / Linux

Requires Python 3.12 (PyTorch does not support Python 3.13 on macOS).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install torch "transformers>=4.30,<5.0" sentencepiece protobuf "numpy<2"
pip install -r requirements.txt
python main.py --input data/job_app_confirmation_emails_anonymized.csv --model deberta
```

### DeBERTa — Windows

PyTorch supports Python 3.12 on Windows — install it from [python.org](https://www.python.org/downloads/) if needed.

```bat
python -m venv venv
venv\Scripts\activate
pip install torch "transformers>=4.30,<5.0" sentencepiece protobuf "numpy<2"
pip install -r requirements.txt
python main.py --input data/job_app_confirmation_emails_anonymized.csv --model deberta
```

### DeBERTa model presets

```bash
# Balanced — recommended (~20 min CPU)
python main.py --input data/... --model deberta --deberta-model balanced

# Fast — for testing (~12 min CPU)
python main.py --input data/... --model deberta --deberta-model fast

# Best accuracy (~60 min CPU)
python main.py --input data/... --model deberta --deberta-model best
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
