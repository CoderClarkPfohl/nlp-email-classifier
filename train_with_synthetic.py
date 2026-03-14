#!/usr/bin/env python3
"""
Train on Synthetic + Real Data, Evaluate Properly
===================================================
This script:
  1. Loads the synthetic dataset (2000 emails with ground-truth labels)
  2. Loads the real dataset (494 emails with rule-based pseudo-labels)
  3. Trains the ensemble on synthetic data
  4. Evaluates on real data (cross-validated)
  5. Trains a combined model on everything for final deployment
  6. Compares accuracy: synthetic-only vs real-only vs combined
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

from utils.preprocessing import clean_email_body, preprocess_for_model
from utils.entity_extraction import extract_entities
from utils.summarizer import summarize_email
from models.rule_labeler import label_email
from models.sentiment import compute_sentiment
from models.svm_classifier import (
    build_tfidf, build_ensemble, oversample_minority, train_and_evaluate
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_preprocess(path, label_col=None):
    """Load CSV and preprocess text."""
    df = pd.read_csv(path).dropna(subset=["email_body"]).reset_index(drop=True)
    df["clean_body"] = df["email_body"].apply(clean_email_body)

    if label_col and label_col in df.columns:
        df["label"] = df[label_col]
    else:
        # Apply rule labeler
        results = df.apply(
            lambda r: label_email(str(r.get("subject", "")),
                                  str(r.get("email_body", ""))),
            axis=1,
        )
        df["label"] = [r[0] for r in results]
        df["label_confidence"] = [r[1] for r in results]

    return df


def evaluate_model(tfidf, ensemble, X_test, y_test, label_set, name=""):
    """Evaluate a trained model and return metrics."""
    preds = ensemble.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, labels=label_set,
                                    zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, preds, labels=label_set)
    report_str = classification_report(y_test, preds, labels=label_set,
                                        zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(report_str)
    print(f"  Accuracy: {acc:.4f}")

    return acc, report, cm, preds


def main():
    print("=" * 60)
    print("  TRAINING WITH SYNTHETIC + REAL DATA")
    print("=" * 60)

    # ── Load datasets ──
    print("\n[1] Loading datasets...")
    df_synth = load_and_preprocess("data/synthetic_emails.csv", label_col="true_label")
    df_real = load_and_preprocess("data/job_app_confirmation_emails_anonymized.csv")

    print(f"  Synthetic: {len(df_synth)} emails")
    print(f"    {dict(Counter(df_synth['label']))}")
    print(f"  Real:      {len(df_real)} emails")
    print(f"    {dict(Counter(df_real['label']))}")

    label_set = sorted(set(df_synth["label"].tolist() + df_real["label"].tolist()))
    print(f"  Labels: {label_set}")

    # ── Experiment 1: Train on REAL only (baseline) ──
    print("\n[2] Experiment 1: Train on REAL data only (5-fold CV)...")
    tfidf_r, ens_r, cv_preds_r, metrics_r = train_and_evaluate(
        df_real["clean_body"].tolist(), df_real["label"].tolist(), n_folds=5
    )
    acc_real_only = metrics_r["mean_cv_accuracy"]

    # ── Experiment 2: Train on SYNTHETIC, test on REAL ──
    print("\n[3] Experiment 2: Train on SYNTHETIC → Test on REAL...")
    tfidf_s = build_tfidf()
    X_synth = tfidf_s.fit_transform(df_synth["clean_body"])
    y_synth = df_synth["label"].tolist()

    # Oversample for balance
    X_synth_os, y_synth_os = oversample_minority(X_synth, y_synth, min_samples=50)

    ens_s = build_ensemble()
    ens_s.fit(X_synth_os, y_synth_os)

    # Test on real data
    X_real_s = tfidf_s.transform(df_real["clean_body"])
    y_real = df_real["label"].tolist()

    acc_synth_on_real, _, cm_synth, preds_synth = evaluate_model(
        tfidf_s, ens_s, X_real_s, y_real, label_set,
        "Trained on SYNTHETIC → Tested on REAL"
    )

    # ── Experiment 3: Train on COMBINED (synthetic + real) ──
    print("\n[4] Experiment 3: Train on COMBINED, 5-fold CV on REAL...")

    # Combine datasets
    df_combined = pd.concat([df_synth, df_real], ignore_index=True)
    print(f"  Combined: {len(df_combined)} emails")
    print(f"    {dict(Counter(df_combined['label']))}")

    # Fit TF-IDF on combined
    tfidf_c = build_tfidf()
    X_combined = tfidf_c.fit_transform(df_combined["clean_body"])
    y_combined = df_combined["label"].tolist()

    # For evaluation: train on synthetic + train_fold(real), test on test_fold(real)
    X_real_c = tfidf_c.transform(df_real["clean_body"])
    X_synth_c = tfidf_c.transform(df_synth["clean_body"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds_combined = np.array([""] * len(df_real), dtype=object)
    fold_accs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real_c, y_real)):
        # Training set = ALL synthetic + train fold of real
        from scipy.sparse import vstack
        X_train = vstack([X_synth_c, X_real_c[train_idx]])
        y_train = y_synth + [y_real[i] for i in train_idx]

        X_train_os, y_train_os = oversample_minority(X_train, y_train, min_samples=50)

        ens_c = build_ensemble()
        ens_c.fit(X_train_os, y_train_os)

        X_test = X_real_c[test_idx]
        preds = ens_c.predict(X_test)
        cv_preds_combined[test_idx] = preds

        fold_acc = accuracy_score([y_real[i] for i in test_idx], preds)
        fold_accs.append(fold_acc)
        print(f"    Fold {fold_idx+1}: {fold_acc:.4f}")

    acc_combined = np.mean(fold_accs)
    report_combined_str = classification_report(
        y_real, cv_preds_combined, labels=label_set, zero_division=0
    )
    cm_combined = confusion_matrix(y_real, cv_preds_combined, labels=label_set)

    print(f"\n{'='*60}")
    print(f"  COMBINED: Train on Synthetic+Real → Test on Real (CV)")
    print(f"{'='*60}")
    print(report_combined_str)
    print(f"  Mean accuracy: {acc_combined:.4f}")

    # ── Train FINAL model on everything ──
    print("\n[5] Training final model on ALL data...")
    X_all_os, y_all_os = oversample_minority(X_combined, y_combined, min_samples=50)
    final_ensemble = build_ensemble()
    final_ensemble.fit(X_all_os, y_all_os)

    # ── Apply final model to real data ──
    final_preds = final_ensemble.predict(X_real_c)
    df_real["final_label"] = final_preds

    # Sentiment, entities, summaries on real data
    print("\n[6] Running NLP pipeline on real data...")
    sentiments = df_real["clean_body"].apply(compute_sentiment)
    df_real["sentiment_label"] = [s["label"] for s in sentiments]
    df_real["sentiment_compound"] = [s["compound"] for s in sentiments]

    entities = df_real.apply(
        lambda r: extract_entities(str(r.get("email_body", "")),
                                    str(r.get("company", ""))),
        axis=1,
    )
    df_real["extracted_role"] = [e["job_role"] for e in entities]
    df_real["contact_person"] = [e["contact_person"] for e in entities]
    df_real["contact_email"] = [e["contact_email"] for e in entities]
    df_real["summary"] = df_real["clean_body"].apply(lambda t: summarize_email(t, 2))

    # ── RESULTS COMPARISON ──
    print("\n" + "=" * 60)
    print("  ACCURACY COMPARISON")
    print("=" * 60)
    print(f"  Majority baseline:         {Counter(y_real).most_common(1)[0][1]/len(y_real)*100:.1f}%")
    print(f"  Real data only (CV):       {acc_real_only*100:.1f}%")
    print(f"  Synthetic → Real:          {acc_synth_on_real*100:.1f}%")
    print(f"  Combined Synth+Real (CV):  {acc_combined*100:.1f}%")
    print(f"  DeBERTa zero-shot (est.):  ~94%")
    print(f"  Fine-tuned DeBERTa (est.): ~97%")

    # ── Plots ──
    print("\n[7] Generating plots...")

    # Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    approaches = {
        "Majority\nbaseline": Counter(y_real).most_common(1)[0][1]/len(y_real)*100,
        "Real only\n(ensemble CV)": acc_real_only * 100,
        "Synthetic →\nReal": acc_synth_on_real * 100,
        "Combined\n(synth+real CV)": acc_combined * 100,
        "DeBERTa\nzero-shot (est.)": 94.0,
        "Fine-tuned\nDeBERTa (est.)": 97.0,
    }
    colors = ["#bdc3c7", "#e67e22", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    bars = ax.bar(approaches.keys(), approaches.values(), color=colors, edgecolor="white")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Effect of Synthetic Training Data on Classification Accuracy")
    ax.set_ylim(50, 100)
    ax.axhline(y=96, color="red", linestyle="--", alpha=0.5, label="Human ceiling (~96%)")
    ax.legend()
    for bar, val in zip(bars, approaches.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()

    # Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, cm, title in [
        (axes[0], cm_synth, "Trained on Synthetic → Tested on Real"),
        (axes[1], cm_combined, "Combined (Synth+Real) → CV on Real"),
    ]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_set, yticklabels=label_set, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices_comparison.png"), dpi=150)
    plt.close()

    # Label distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = pd.Series(cv_preds_combined).value_counts()
    axes[0].barh(counts.index, counts.values, color=sns.color_palette("Set2", len(counts)))
    axes[0].set_title("Predicted Labels (Combined Model)")
    axes[0].set_xlabel("Count")

    sent_by_label = df_real.groupby("final_label")["sentiment_compound"].mean().sort_values()
    bar_colors = ["#e74c3c" if v < -0.1 else "#2ecc71" if v > 0.1 else "#95a5a6"
                  for v in sent_by_label.values]
    axes[1].barh(sent_by_label.index, sent_by_label.values, color=bar_colors)
    axes[1].set_title("Avg Sentiment by Predicted Category")
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "label_distribution.png"), dpi=150)
    plt.close()

    # ── Save final CSV ──
    output_cols = [
        "company", "subject", "date_only", "email_body",
        "label", "final_label",
        "sentiment_label", "sentiment_compound",
        "extracted_role", "contact_person", "contact_email", "summary",
    ]
    output_cols = [c for c in output_cols if c in df_real.columns]
    df_real[output_cols].to_csv(
        os.path.join(RESULTS_DIR, "classified_emails.csv"), index=False
    )

    # Save classification report
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write("COMBINED MODEL (Synthetic + Real) Classification Report\n")
        f.write("Evaluated via 5-fold CV on real data\n")
        f.write("=" * 60 + "\n")
        f.write(report_combined_str)
        f.write(f"\nMean accuracy: {acc_combined:.4f}\n")
        f.write(f"\nComparison:\n")
        for name, val in approaches.items():
            f.write(f"  {name.replace(chr(10), ' '):30s} {val:.1f}%\n")

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
