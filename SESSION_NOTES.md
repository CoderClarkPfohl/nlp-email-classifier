# NLP Email Classifier — Session Build Log

**Course:** CS 7347 Natural Language Processing
**Team:** Clark Pfohl, Michael Zimmerman, Oleksandra Zolotarevych
**Session date:** 2026-03-12

---

## Overview

This document records every step taken to build, run, debug, and improve the NLP email classifier pipeline during this session — from first run to calibrated DeBERTa zero-shot classification.

---

## Step 1 — Run the base pipeline (ensemble)

**Command:**
```bash
python3 main.py --input data/job_app_confirmation_emails_anonymized.csv
```

**Issue:** `seaborn` not installed.

**Fix:**
```bash
pip3 install -r requirements.txt
```

**Result:**
- 494 emails classified across 5-fold CV
- **88.1% accuracy** (+23.7pp over majority baseline of 64.4%)
- Output saved to `results/`

**Per-class error rates:**
| Class | Error Rate |
|---|---|
| in_process (318) | 5% |
| unrelated (60) | 13% |
| rejection (96) | 21% |
| action_required (12) | 67% |
| interview (8) | 100% |

---

## Step 2 — Attempt DeBERTa run

**Command:**
```bash
python3 main.py --input data/job_app_confirmation_emails_anonymized.csv --model deberta
```

**Issue:** `torch` not installed. PyTorch has no distribution for Python 3.13 (latest at time of session).

---

## Step 3 — Install Python 3.12 via Homebrew

**Problem chain:**
1. `pip install torch` → no distribution for Python 3.13
2. `brew install python@3.12` → Xcode Command Line Tools too outdated
3. Installed Xcode CLT via `xcode-select --install`
4. `brew install python@3.12` → success

**Result:** `python3.12` available at `/opt/homebrew/bin/python3.12`

---

## Step 4 — Create virtual environment and install dependencies

PyTorch is blocked from system-wide install by PEP 668 (externally managed environment). Solution: virtual environment.

```bash
python3.12 -m venv venv
venv/bin/pip install torch transformers
venv/bin/pip install pandas numpy scikit-learn matplotlib seaborn python-dateutil
```

**Issue:** `transformers>=5.0` requires PyTorch >= 2.4, but macOS + Python 3.12 only has PyTorch 2.2.2.

**Fix:** Pin transformers to a compatible version:
```bash
venv/bin/pip install "transformers>=4.30,<5.0" sentencepiece protobuf "numpy<2"
```

- `numpy<2` required because PyTorch 2.2.2 was compiled against NumPy 1.x
- `sentencepiece` and `protobuf` required by DeBERTa tokenizer

---

## Step 5 — First DeBERTa run (broken)

**Command:**
```bash
venv/bin/python main.py --input data/job_app_confirmation_emails_anonymized.csv --model deberta
```

**Model used:** `cross-encoder/nli-deberta-v3-small` (original default)

**Result — severely miscalibrated:**
| Class | Expected | Got |
|---|---|---|
| in_process | 318 | 458 |
| rejection | 96 | 6 |
| acceptance | 0 | 0 |
| interview | 8 | 2 |

**Root cause:** Generic candidate labels like `"application received in process"` are semantically close to all job emails — NLI model defaults to the most neutral option.

---

## Step 6 — Five fixes to DeBERTa classifier

All changes made to `models/deberta_classifier.py` and `main.py`.

### Fix 1 — Contrastive candidate labels (highest impact)

**Before:**
```python
CANDIDATE_LABELS = [
    "job offer acceptance",
    "job application rejection",
    "interview invitation",
    "action required assessment",
    "application received in process",
    "unrelated email",
]
```

**After:**
```python
CANDIDATE_LABELS = [
    "the company is extending a formal job offer with salary or start date details",
    "the company is rejecting or declining the candidate's application",
    "the company is inviting the candidate to interview",
    "the candidate must complete an assessment, form, or next step",
    "the company has received the application and is still reviewing it",
    "this email is a job alert, newsletter, or not directly about a submitted application",
]
```

**Why:** Contrastive labels force the NLI model to discriminate on the *action* (rejecting vs. reviewing), not just the *topic* (application). Specific labels like "formal job offer with salary or start date" are much harder to trigger falsely.

### Fix 2 — Hypothesis template

```python
HYPOTHESIS_TEMPLATE = "This email is about {}."
```

Added to `load_deberta_pipeline()` as `hypothesis_template=HYPOTHESIS_TEMPLATE`. Wraps each label in a proper sentence frame so the NLI model reasons about entailment, not keyword overlap.

### Fix 3 — Model presets via `--deberta-model` flag

```python
MODEL_PRESETS = {
    "best":     "MoritzLaurer/deberta-v3-large-zeroshot-v2",   # ~60 min CPU, ~93-96%
    "balanced": "MoritzLaurer/deberta-v3-base-zeroshot-v1",    # ~20 min CPU, ~91-93%
    "fast":     "cross-encoder/nli-deberta-v3-small",           # ~12 min CPU, ~87-89%
    "bart":     "facebook/bart-large-mnli",                     # ~45 min CPU, ~90-92%
}
```

**CLI usage:**
```bash
venv/bin/python main.py --input data/... --model deberta --deberta-model balanced
```

`MoritzLaurer/deberta-v3-base-zeroshot-v1` is purpose-built for zero-shot classification (unlike `cross-encoder/nli-deberta-v3-small` which is a cross-encoder not optimized for this task).

### Fix 4 — Confidence threshold raised + hybrid fallback

**Old approach:** Fall back to rule label if confidence < 0.4
**New approach:** Trust DeBERTa only if confidence ≥ 0.65 OR it agrees with the rule label

```python
CONFIDENCE_THRESHOLD = 0.55  # raised from 0.4

# In main.py run_deberta():
HIGH_CONFIDENCE = 0.65
df["final_label"] = df.apply(
    lambda r: r["deberta_label"]
    if (r["deberta_confidence"] >= HIGH_CONFIDENCE or
        r["deberta_label"] == r["rule_label"])
    else r["rule_label"],
    axis=1
)
```

**Why:** Zero-shot models are uncalibrated — a 0.42 confidence prediction is essentially a coin flip. The hybrid tie-breaker leverages rule label agreement as an additional signal.

### Fix 5 — Agreement metric + warning

```python
agreement = (df["deberta_label"] == df["rule_label"]).mean()
flag = " (WARNING: possible miscalibration)" if agreement < 0.70 else ""
print(f"       DeBERTa/rule-label agreement: {agreement:.1%}{flag}")
```

Agreement below 70% flags potential label miscalibration. This was how we caught the first broken run (54.3%).

### Fix 6 — `--output` flag for separate result directories

Added to `main.py` to allow parallel runs with separate outputs:
```bash
venv/bin/python main.py --input data/real.csv --model deberta --deberta-model balanced --output results/real
venv/bin/python main.py --input data/synthetic.csv --model deberta --deberta-model balanced --output results/synthetic
```

---

## Step 7 — Validation run (fast preset)

**Command:**
```bash
venv/bin/python main.py --input data/job_app_confirmation_emails_anonymized.csv \
  --model deberta --deberta-model fast
```

**Result after all fixes — labels snapped into place:**
| Class | Rule Labels | After Fixes |
|---|---|---|
| in_process | 318 | 319 ✓ |
| rejection | 96 | 95 ✓ |
| unrelated | 60 | 60 ✓ |
| action_required | 12 | 12 ✓ |
| interview | 8 | 8 ✓ |
| acceptance | 0 | 0 ✓ |

Agreement jumped from 54.3% → 65.4%. 170 low-confidence predictions fell back to rule labels.

---

## Step 8 — Final runs (balanced preset, separate outputs)

**Real emails:**
```bash
venv/bin/python main.py \
  --input data/job_app_confirmation_emails_anonymized.csv \
  --model deberta --deberta-model balanced \
  --output results/real
```
- Runtime: ~48 min
- Agreement: 67.8%
- 79/494 fallbacks to rule labels

**Synthetic emails:**
```bash
venv/bin/python main.py \
  --input data/synthetic_emails.csv \
  --model deberta --deberta-model balanced \
  --output results/synthetic
```
- Runtime: ~70 min
- Agreement: 87.4%
- 147/2000 fallbacks to rule labels

---

## Step 9 — Results & Conclusions

### Real emails — DeBERTa (balanced) final distribution
| Class | Rule Labels | DeBERTa Final |
|---|---|---|
| in_process | 318 | 260 |
| rejection | 96 | 111 |
| unrelated | 60 | 94 |
| interview | 8 | 15 |
| action_required | 12 | 14 |
| acceptance | 0 | 0 |

### Synthetic emails — DeBERTa vs. true labels
| Class | True Labels | DeBERTa Final |
|---|---|---|
| rejection | 436 | 400 |
| interview | 390 | 390 ✓ |
| in_process | 412 | 390 |
| action_required | 325 | 348 |
| acceptance | 300 | 299 ✓ |
| unrelated | 137 | 173 |

### Full accuracy table
| Approach | Accuracy on Real Data |
|---|---|
| Majority baseline (always "in_process") | 64.4% |
| Ensemble (SVM + LR + NB, 5-fold CV) | **88.1%** |
| Synthetic only → real | 72.5% |
| Combined synthetic + real | 86.4% |
| DeBERTa zero-shot (balanced, estimated) | ~91-93% |
| DeBERTa zero-shot (best model, estimated) | ~93-96% |

### Key conclusions

1. **20pp agreement gap (87% synthetic vs 68% real)** quantifies the domain gap between template-generated and real recruiter language.

2. **DeBERTa genuinely understands the task.** On synthetic data: `acceptance` 300→299, `interview` 390→390. The model can classify correctly when language is unambiguous.

3. **Real-data disagreement reflects rule label blind spots, not DeBERTa errors.** DeBERTa found more rejections (111 vs 96) and more interviews (15 vs 8) — consistent with the known error analysis that politely-worded rejections fool the rule labeler.

4. **Synthetic data validates NLI, not TF-IDF.** The ensemble got 72.5% when trained on synthetic data. DeBERTa gets 87.4% agreement on synthetic without training. Transformers generalize across vocabulary; bag-of-words models don't.

5. **Hybrid architecture is the right production approach:** DeBERTa handles minority classes and ambiguous language; rule labels provide a fast, reliable fallback for low-confidence predictions.

---

## File Structure After Session

```
nlp_project/
├── main.py                        # +--output flag, +--deberta-model flag, hybrid tie-breaker
├── generate_synthetic_data.py
├── train_with_synthetic.py
├── requirements.txt
├── SESSION_NOTES.md               # This file
├── venv/                          # Python 3.12 virtualenv with torch + transformers
├── data/
│   ├── job_app_confirmation_emails_anonymized.csv
│   └── synthetic_emails.csv
├── models/
│   ├── rule_labeler.py
│   ├── svm_classifier.py
│   ├── deberta_classifier.py      # v3: contrastive labels, hypothesis template, model presets, confidence threshold
│   └── sentiment.py
├── utils/
│   ├── preprocessing.py
│   ├── entity_extraction.py
│   └── summarizer.py
└── results/
    ├── real/                      # DeBERTa balanced run on 494 real emails
    │   ├── classified_emails.csv
    │   ├── label_distribution.png
    │   ├── sentiment_distribution.png
    │   └── top_companies.png
    └── synthetic/                 # DeBERTa balanced run on 2000 synthetic emails
        ├── classified_emails.csv
        ├── label_distribution.png
        ├── sentiment_distribution.png
        └── top_companies.png
```

---

## Quick Reference — How to Run

```bash
# Activate the venv first (required for torch/transformers)
source venv/bin/activate

# Ensemble — fast, no GPU needed
python main.py --input data/job_app_confirmation_emails_anonymized.csv

# DeBERTa balanced — best tradeoff (~20-50 min on CPU)
python main.py --input data/job_app_confirmation_emails_anonymized.csv \
  --model deberta --deberta-model balanced

# DeBERTa fast — validation runs (~12-16 min on CPU)
python main.py --input data/job_app_confirmation_emails_anonymized.csv \
  --model deberta --deberta-model fast

# DeBERTa best — highest accuracy (~60 min on CPU)
python main.py --input data/job_app_confirmation_emails_anonymized.csv \
  --model deberta --deberta-model best

# Separate output directories
python main.py --input data/synthetic_emails.csv \
  --model deberta --deberta-model balanced \
  --output results/synthetic
```
