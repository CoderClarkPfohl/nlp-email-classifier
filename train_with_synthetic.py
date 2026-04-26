#!/usr/bin/env python3
"""
Train on Synthetic + Real Data, Evaluate Properly
===================================================
This script:
  1. Loads the synthetic dataset (2000 emails, uses ground-truth true_label)
  2. Loads the real dataset (494 emails with rule-based pseudo-labels)
  3. Trains the ensemble on synthetic data → tests on real
  4. Trains on real only (5-fold CV baseline)
  5. Trains on combined synthetic + real → CV on real
  6. Compares accuracy across all three experiments
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

from utils.preprocessing import clean_email_body
from models.rule_labeler import label_email
from models.svm_classifier import (
    build_tfidf, build_ensemble, oversample_minority, train_and_evaluate
)
from utils.excel_export import export_to_excel
from main import run_nlp_enrichment  # shared enrichment pipeline

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_preprocess(path: str, label_col: str = None) -> pd.DataFrame:
    """
    Load CSV, clean email bodies, and assign labels.

    If label_col is given and present in the CSV (e.g. 'true_label' on synthetic
    data), use it directly — no rule labeler needed.  Otherwise fall back to the
    rule labeler so real emails still get pseudo-labels.
    """
    df = pd.read_csv(path).dropna(subset=["email_body"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["email_body"]).reset_index(drop=True)
    df["clean_body"] = df["email_body"].apply(clean_email_body)

    if label_col and label_col in df.columns:
        df["label"] = df[label_col]
    else:
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
    cm = confusion_matrix(y_test, preds, labels=label_set)
    report_str = classification_report(y_test, preds, labels=label_set, zero_division=0)
    report_dict = classification_report(y_test, preds, labels=label_set,
                                        zero_division=0, output_dict=True)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(report_str)
    print(f"  Accuracy: {acc:.4f}")

    return acc, report_dict, cm, preds


def main():
    print("=" * 60)
    print("  TRAINING WITH SYNTHETIC + REAL DATA")
    print("=" * 60)

    # ── [1] Load datasets ──
    print("\n[1] Loading datasets...")
    # Use true_label from synthetic CSV — clean ground truth, no rule labeler noise
    df_synth = load_and_preprocess("data/synthetic_emails.csv", label_col="true_label")
    df_real = load_and_preprocess("data/job_app_confirmation_emails_anonymized.csv")

    print(f"  Synthetic: {len(df_synth)} emails (using true_label ground truth)")
    print(f"    {dict(sorted(Counter(df_synth['label']).items()))}")
    print(f"  Real:      {len(df_real)} emails (rule-labeler pseudo-labels)")
    print(f"    {dict(sorted(Counter(df_real['label']).items()))}")

    label_set = sorted(set(df_synth["label"].tolist() + df_real["label"].tolist()))
    print(f"  Labels: {label_set}")

    y_real = df_real["label"].tolist()
    y_synth = df_synth["label"].tolist()

    # ── [2] Experiment 1: Real only (5-fold CV baseline) ──
    print("\n[2] Experiment 1: Train on REAL data only (5-fold CV)...")
    _, _, cv_preds_r, _, metrics_r = train_and_evaluate(
        df_real["clean_body"].tolist(), y_real, n_folds=5
    )
    acc_real_only = metrics_r["mean_cv_accuracy"]

    # ── [3] Experiment 2: Synthetic → Real ──
    print("\n[3] Experiment 2: Train on SYNTHETIC → Test on REAL...")
    tfidf_s = build_tfidf()
    X_synth = tfidf_s.fit_transform(df_synth["clean_body"])
    X_synth_os, y_synth_os = oversample_minority(X_synth, y_synth, min_samples=50)

    ens_s = build_ensemble()
    ens_s.fit(X_synth_os, y_synth_os)

    X_real_s = tfidf_s.transform(df_real["clean_body"])
    acc_synth_on_real, _, cm_synth, _ = evaluate_model(
        tfidf_s, ens_s, X_real_s, y_real, label_set,
        "Trained on SYNTHETIC → Tested on REAL"
    )

    # ── [4] Experiment 3: Combined, 5-fold CV on real portion ──
    print("\n[4] Experiment 3: Train on COMBINED, 5-fold CV on REAL...")
    df_combined = pd.concat([df_synth, df_real], ignore_index=True)
    y_combined = df_combined["label"].tolist()
    print(f"  Combined: {len(df_combined)} emails")

    tfidf_c = build_tfidf()
    X_combined = tfidf_c.fit_transform(df_combined["clean_body"])
    X_real_c = tfidf_c.transform(df_real["clean_body"])
    X_synth_c = tfidf_c.transform(df_synth["clean_body"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds_combined = np.array([""] * len(df_real), dtype=object)
    fold_accs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real_c, y_real)):
        from scipy.sparse import vstack
        # Train on ALL synthetic (true_label ground truth) + real train fold
        X_train = vstack([X_synth_c, X_real_c[train_idx]])
        y_train = y_synth + [y_real[i] for i in train_idx]

        X_train_os, y_train_os = oversample_minority(X_train, y_train, min_samples=50)

        ens_c = build_ensemble()
        ens_c.fit(X_train_os, y_train_os)

        preds = ens_c.predict(X_real_c[test_idx])
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

    # ── [5] Train final model on all data ──
    print("\n[5] Training final model on ALL data...")
    X_all_os, y_all_os = oversample_minority(X_combined, y_combined, min_samples=50)
    final_ensemble = build_ensemble()
    final_ensemble.fit(X_all_os, y_all_os)

    final_preds = final_ensemble.predict(X_real_c)
    df_real["final_label"] = final_preds

    # ── [6] NLP enrichment on real data (shared pipeline) ──
    print("\n[6] Running NLP pipeline on real data...")
    df_real = run_nlp_enrichment(df_real)

    # ── Results comparison ──
    majority_pct = Counter(y_real).most_common(1)[0][1] / len(y_real) * 100
    print("\n" + "=" * 60)
    print("  ACCURACY COMPARISON")
    print("=" * 60)
    print(f"  Majority baseline:         {majority_pct:.1f}%")
    print(f"  Real data only (CV):       {acc_real_only*100:.1f}%")
    print(f"  Synthetic → Real:          {acc_synth_on_real*100:.1f}%")
    print(f"  Combined Synth+Real (CV):  {acc_combined*100:.1f}%")
    print(f"  DeBERTa zero-shot (est.):  ~94%")
    print(f"  Fine-tuned DeBERTa (est.): ~97%")

    # ── [7] Plots ──
    print("\n[7] Generating plots...")
    approaches = {
        "Majority\nbaseline": majority_pct,
        "Real only\n(ensemble CV)": acc_real_only * 100,
        "Synthetic →\nReal": acc_synth_on_real * 100,
        "Combined\n(synth+real CV)": acc_combined * 100,
        "DeBERTa\nzero-shot (est.)": 94.0,
        "Fine-tuned\nDeBERTa (est.)": 97.0,
    }
    colors = ["#bdc3c7", "#e67e22", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(approaches.keys(), approaches.values(), color=colors, edgecolor="white")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Effect of Synthetic Training Data on Classification Accuracy")
    ax.set_ylim(max(0, majority_pct - 15), 100)
    ax.axhline(y=96, color="red", linestyle="--", alpha=0.5, label="Human ceiling (~96%)")
    ax.legend()
    for bar, val in zip(bars, approaches.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_item, cm, title in [
        (axes[0], cm_synth, "Trained on Synthetic → Tested on Real"),
        (axes[1], cm_combined, "Combined (Synth+Real) → CV on Real"),
    ]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_set, yticklabels=label_set, ax=ax_item)
        ax_item.set_xlabel("Predicted")
        ax_item.set_ylabel("True")
        ax_item.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices_comparison.png"), dpi=150)
    plt.close()

    if "sentiment_compound" in df_real.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        counts = pd.Series(cv_preds_combined).value_counts()
        axes[0].barh(counts.index, counts.values,
                     color=sns.color_palette("Set2", len(counts)))
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

    # ── Save output ──
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
    export_to_excel(
        df_real[output_cols],
        os.path.join(RESULTS_DIR, "classified_emails.xlsx"),
        label_column="final_label",
    )
    print(f"  Color-coded Excel saved: {os.path.join(RESULTS_DIR, 'classified_emails.xlsx')}")

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write("COMBINED MODEL (Synthetic + Real) Classification Report\n")
        f.write("Synthetic data uses ground-truth true_label (not rule labeler)\n")
        f.write("Evaluated via 5-fold CV on real data\n")
        f.write("=" * 60 + "\n")
        f.write(report_combined_str)
        f.write(f"\nMean accuracy: {acc_combined:.4f}\n\n")
        f.write("Comparison:\n")
        for name, val in approaches.items():
            f.write(f"  {name.replace(chr(10), ' '):35s} {val:.1f}%\n")

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
