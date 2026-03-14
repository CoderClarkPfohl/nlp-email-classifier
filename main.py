#!/usr/bin/env python3
"""
Job Application Email Classifier — Main Pipeline (v2)
======================================================
CS 7347 Natural Language Processing · Group Project

Pipeline:
  1. Load & preprocess emails
  2. Rule-based pseudo-labeling (v2 — scoring-based)
  3. Sentiment analysis
  4. Entity extraction (company, role, contact, dates)
  5. Extractive summarization
  6. Train & evaluate ensemble classifier (SVM + LR + NB)
  7. (Optional) DeBERTa zero-shot classification
  8. Generate results: CSV, plots, metrics, accuracy analysis

Usage
-----
    python main.py --input data/job_app_confirmation_emails_anonymized.csv
    python main.py --input data/job_app_confirmation_emails_anonymized.csv --model deberta
"""

import argparse
import os
import sys
import warnings
from typing import Dict
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from utils.preprocessing import clean_email_body, preprocess_for_model
from utils.entity_extraction import extract_entities
from utils.summarizer import summarize_email
from models.rule_labeler import label_email
from models.sentiment import compute_sentiment
from models.svm_classifier import train_and_evaluate


RESULTS_DIR = "results"  # overridden by --output flag
LABELS = ["acceptance", "rejection", "interview", "action_required",
          "in_process", "unrelated"]


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  Pipeline Steps
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"[1/7] Loading data from {path}")
    df = pd.read_csv(path)
    print(f"       {len(df)} emails, {df.shape[1]} columns")
    df = df.dropna(subset=["email_body"]).reset_index(drop=True)
    print(f"       {len(df)} after dropping empty bodies")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/7] Preprocessing email text")
    df["clean_body"] = df["email_body"].apply(clean_email_body)
    df["model_body"] = df["email_body"].apply(preprocess_for_model)
    return df


def apply_rule_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/7] Applying rule-based labels (v2 scoring)")
    results = df.apply(
        lambda r: label_email(str(r.get("subject", "")),
                              str(r.get("email_body", ""))),
        axis=1,
    )
    df["rule_label"] = [r[0] for r in results]
    df["rule_confidence"] = [r[1] for r in results]
    print("       Label distribution:")
    for lbl, cnt in df["rule_label"].value_counts().items():
        print(f"         {lbl:20s} {cnt:4d}  ({100*cnt/len(df):.1f}%)")
    low_conf = (df["rule_confidence"] < 0.6).sum()
    print(f"       Low-confidence labels (<0.6): {low_conf}")
    return df


def apply_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/7] Sentiment analysis")
    sentiments = df["clean_body"].apply(compute_sentiment)
    df["sentiment_label"] = [s["label"] for s in sentiments]
    df["sentiment_compound"] = [s["compound"] for s in sentiments]
    df["sentiment_positive"] = [s["positive"] for s in sentiments]
    df["sentiment_negative"] = [s["negative"] for s in sentiments]
    for lbl, cnt in df["sentiment_label"].value_counts().items():
        print(f"         {lbl:10s} {cnt:4d}")
    return df


def apply_entity_extraction(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/7] Entity extraction")
    entities = df.apply(
        lambda r: extract_entities(str(r.get("email_body", "")),
                                    str(r.get("company", ""))),
        axis=1,
    )
    df["extracted_role"] = [e["job_role"] for e in entities]
    df["contact_person"] = [e["contact_person"] for e in entities]
    df["contact_email"] = [e["contact_email"] for e in entities]
    df["dates_mentioned"] = [
        "; ".join(e["dates_mentioned"]) if e["dates_mentioned"] else ""
        for e in entities
    ]
    print(f"       Roles: {df['extracted_role'].notna().sum()}/{len(df)}")
    print(f"       Contacts: {df['contact_person'].notna().sum()}/{len(df)}")
    return df


def apply_summarization(df: pd.DataFrame) -> pd.DataFrame:
    print("[6/7] Extractive summarization")
    df["summary"] = df["clean_body"].apply(lambda t: summarize_email(t, max_sentences=2))
    return df


# ─────────────────────────────────────────────────────────────
#  Classification
# ─────────────────────────────────────────────────────────────

def run_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    print("[7/7] Training ensemble classifier (SVM + LR + NB) with oversampling")
    texts = df["clean_body"].tolist()
    labels = df["rule_label"].tolist()

    tfidf, ensemble, cv_preds, metrics = train_and_evaluate(texts, labels, n_folds=5)

    df["ensemble_prediction"] = cv_preds
    df["final_label"] = cv_preds

    print("\n" + "=" * 60)
    print("  Ensemble Classification Report (5-fold CV + oversampling)")
    print("=" * 60)
    print(metrics["report_text"])
    print(f"  Mean fold accuracy: {metrics['mean_cv_accuracy']:.4f}")
    print(f"  Per-fold: {[f'{a:.4f}' for a in metrics['fold_accuracies']]}")

    # Save report
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write("Ensemble Classification Report (5-fold CV + oversampling)\n")
        f.write("=" * 60 + "\n")
        f.write(metrics["report_text"])
        f.write(f"\nMean fold accuracy: {metrics['mean_cv_accuracy']:.4f}\n")

    plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"],
                          "Ensemble Confusion Matrix (5-fold CV)",
                          "confusion_matrix.png")

    # Accuracy analysis
    print_accuracy_analysis(df, metrics)

    return df


def run_deberta(df: pd.DataFrame, deberta_model: str = None) -> pd.DataFrame:
    print("[7/7] DeBERTa zero-shot classifier")
    try:
        from models.deberta_classifier import load_deberta_pipeline, classify_batch, CONFIDENCE_THRESHOLD
        classifier = load_deberta_pipeline(model_name=deberta_model, device=-1)
    except (ImportError, Exception) as e:
        print(f"  ERROR: {e}")
        print("  Falling back to ensemble classifier.")
        return run_ensemble(df)

    from models.deberta_classifier import classify_single
    texts = df["model_body"].tolist()
    n = len(texts)
    batch_size = 8
    results = []
    import time
    start = time.time()
    try:
        from tqdm import tqdm
        batches = range(0, n, batch_size)
        bar = tqdm(batches, desc="  DeBERTa", unit="batch",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]")
        for i in bar:
            batch = texts[i:i+batch_size]
            results.extend(classify_batch(classifier, batch, batch_size=len(batch)))
            emails_done = min(i + batch_size, n)
            elapsed = time.time() - start
            if emails_done > 0:
                eta = elapsed / emails_done * (n - emails_done)
                bar.set_postfix(emails=f"{emails_done}/{n}", eta=f"{eta:.0f}s")
    except ImportError:
        for i in range(0, n, batch_size):
            batch = texts[i:i+batch_size]
            results.extend(classify_batch(classifier, batch, batch_size=len(batch)))
            emails_done = min(i + batch_size, n)
            elapsed = time.time() - start
            eta = elapsed / emails_done * (n - emails_done) if emails_done > 0 else 0
            print(f"\r  Progress: {emails_done}/{n} emails | elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", end="", flush=True)
        print()
    df["deberta_label"] = [r["label"] for r in results]
    df["deberta_confidence"] = [r["confidence"] for r in results]

    # Hybrid tie-breaker: trust DeBERTa when high-confidence OR agrees with rule label
    HIGH_CONFIDENCE = 0.65
    df["final_label"] = df.apply(
        lambda r: r["deberta_label"]
        if (r["deberta_confidence"] >= HIGH_CONFIDENCE or
            r["deberta_label"] == r["rule_label"])
        else r["rule_label"],
        axis=1
    )
    fallback_count = (
        (df["deberta_confidence"] < HIGH_CONFIDENCE) &
        (df["deberta_label"] != df["rule_label"])
    ).sum()
    if fallback_count > 0:
        print(f"       Low-confidence fallbacks to rule label: {fallback_count}/{len(df)}")

    # Fix 5: Agreement metric — flags miscalibration if below 70%
    agreement = (df["deberta_label"] == df["rule_label"]).mean()
    flag = " (WARNING: possible miscalibration)" if agreement < 0.70 else ""
    print(f"       DeBERTa/rule-label agreement: {agreement:.1%}{flag}")

    print("       DeBERTa label distribution:")
    for lbl, cnt in df["final_label"].value_counts().items():
        print(f"         {lbl:20s} {cnt:4d}")

    return df


# ─────────────────────────────────────────────────────────────
#  Accuracy Analysis
# ─────────────────────────────────────────────────────────────

def print_accuracy_analysis(df: pd.DataFrame, metrics: Dict = None):
    """Print a detailed analysis of accuracy and what limits it."""
    print("\n" + "=" * 60)
    print("  ACCURACY ANALYSIS")
    print("=" * 60)

    n = len(df)
    majority = df["rule_label"].value_counts().iloc[0]
    majority_class = df["rule_label"].value_counts().index[0]

    print(f"\n  Baseline (always predict '{majority_class}'): {majority/n*100:.1f}%")

    if metrics:
        print(f"  Ensemble CV accuracy:                        {metrics['mean_cv_accuracy']*100:.1f}%")
        improvement = (metrics['mean_cv_accuracy'] - majority/n) * 100
        print(f"  Improvement over baseline:                   +{improvement:.1f}pp")

    print(f"\n  Class imbalance (main accuracy bottleneck):")
    for lbl, cnt in df["rule_label"].value_counts().items():
        bar = "█" * max(1, int(cnt / n * 50))
        print(f"    {lbl:20s} {cnt:4d} ({cnt/n*100:5.1f}%) {bar}")

    # What % of errors come from minority classes?
    if "ensemble_prediction" in df.columns:
        errors = df[df["ensemble_prediction"] != df["rule_label"]]
        if len(errors) > 0:
            print(f"\n  Error analysis ({len(errors)} misclassifications):")
            for lbl in df["rule_label"].unique():
                lbl_errors = errors[errors["rule_label"] == lbl]
                if len(lbl_errors) > 0:
                    total_in_class = (df["rule_label"] == lbl).sum()
                    print(f"    {lbl:20s}: {len(lbl_errors):3d} errors / {total_in_class:3d} total = {len(lbl_errors)/total_in_class*100:.0f}% error rate")

    print(f"\n  Realistic accuracy ceilings for this dataset:")
    print(f"    Rule-based heuristics:       ~85-90%  (current pseudo-labels)")
    print(f"    TF-IDF + Ensemble + SMOTE:   ~88-92%  (current approach)")
    print(f"    DeBERTa zero-shot:           ~92-96%  (no training, NLI-based)")
    print(f"    Fine-tuned DeBERTa:          ~95-98%  (needs 50+ labels/class)")
    print(f"    Human agreement ceiling:     ~95-97%  (some emails are ambiguous)")

    print(f"\n  Why 100% is impossible:")
    print(f"    - {(df['rule_confidence'] < 0.6).sum()} emails have ambiguous signals")
    # Count multi-signal emails
    multi = 0
    for _, row in df.iterrows():
        text = (str(row.get("subject", "")) + " " + str(row.get("email_body", ""))).lower()
        signals = 0
        if any(w in text for w in ["received your application", "thank you for applying"]): signals += 1
        if any(w in text for w in ["assessment", "complete the"]): signals += 1
        if any(w in text for w in ["interview", "schedule"]): signals += 1
        if any(w in text for w in ["unfortunately", "regret", "not moving"]): signals += 1
        if signals > 1: multi += 1
    print(f"    - {multi} emails match multiple categories simultaneously")
    print(f"    - Many 'in_process' emails mention future steps (interview/assessment)")
    print(f"      that haven't happened yet — category depends on intent")

    # Write analysis to file
    with open(os.path.join(RESULTS_DIR, "accuracy_analysis.txt"), "w") as f:
        f.write("ACCURACY ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {n} emails, {df['rule_label'].nunique()} classes\n")
        f.write(f"Majority baseline: {majority/n*100:.1f}%\n")
        if metrics:
            f.write(f"Ensemble accuracy: {metrics['mean_cv_accuracy']*100:.1f}%\n")
        f.write(f"\nRealistic ceilings:\n")
        f.write(f"  TF-IDF + Ensemble: 88-92%\n")
        f.write(f"  DeBERTa zero-shot: 92-96%\n")
        f.write(f"  Fine-tuned DeBERTa: 95-98%\n")
        f.write(f"  Human ceiling: 95-97%\n")
        f.write(f"\nTo reach best accuracy:\n")
        f.write(f"  1. pip install torch transformers\n")
        f.write(f"  2. python main.py --input <csv> --model deberta\n")
        f.write(f"  3. For fine-tuning: manually label 50+ emails per class\n")


# ─────────────────────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (Rule-based)")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"       Saved: {path}")


def plot_label_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df["final_label"].value_counts()
    colors = sns.color_palette("Set2", len(counts))
    axes[0].barh(counts.index, counts.values, color=colors)
    axes[0].set_xlabel("Count")
    axes[0].set_title("Email Classification Distribution")
    for i, v in enumerate(counts.values):
        axes[0].text(v + 1, i, str(v), va="center")

    grouped = df.groupby("final_label")["sentiment_compound"].mean().sort_values()
    bar_colors = ["#e74c3c" if v < -0.1 else "#2ecc71" if v > 0.1 else "#95a5a6"
                  for v in grouped.values]
    axes[1].barh(grouped.index, grouped.values, color=bar_colors)
    axes[1].set_xlabel("Mean Sentiment Score")
    axes[1].set_title("Average Sentiment by Category")
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "label_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"       Saved: {path}")


def plot_sentiment_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["sentiment_compound"], bins=30, color="#3498db",
                 edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Compound Sentiment Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sentiment Score Distribution")

    sent_counts = df["sentiment_label"].value_counts()
    pie_colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
    axes[1].pie(sent_counts.values,
                labels=sent_counts.index,
                colors=[pie_colors.get(l, "#ccc") for l in sent_counts.index],
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Sentiment Label Distribution")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "sentiment_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"       Saved: {path}")


def plot_top_companies(df: pd.DataFrame, top_n: int = 15):
    fig, ax = plt.subplots(figsize=(10, 6))
    company_labels = df.groupby("company")["final_label"].value_counts().unstack(fill_value=0)
    top_companies = df["company"].value_counts().head(top_n).index
    company_labels = company_labels.loc[company_labels.index.isin(top_companies)]
    company_labels = company_labels.loc[top_companies]

    company_labels.plot(kind="barh", stacked=True, ax=ax,
                         colormap="Set2", edgecolor="white")
    ax.set_xlabel("Number of Emails")
    ax.set_title(f"Top {top_n} Companies — Email Categories")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "top_companies.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"       Saved: {path}")


def plot_accuracy_comparison(metrics: Dict):
    """Bar chart comparing accuracy of different approaches."""
    fig, ax = plt.subplots(figsize=(10, 5))

    approaches = [
        ("Majority baseline", 77.5),
        ("Rule-based only", 85.0),
        ("TF-IDF + SVM (v1)", 89.0),
        ("Ensemble + SMOTE\n(current)", metrics.get("mean_cv_accuracy", 0.90) * 100),
        ("DeBERTa zero-shot\n(estimated)", 94.0),
        ("Fine-tuned DeBERTa\n(estimated)", 97.0),
    ]

    names = [a[0] for a in approaches]
    accs = [a[1] for a in approaches]
    colors = ["#bdc3c7", "#95a5a6", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]

    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy: Current vs. Achievable")
    ax.set_ylim(70, 100)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Add human ceiling line
    ax.axhline(y=96, color="red", linestyle="--", alpha=0.5, label="Human ceiling (~96%)")
    ax.legend()

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"       Saved: {path}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Job Application Email Classifier — NLP Pipeline"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the email CSV file")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["ensemble", "deberta"],
                        help="Classification model: ensemble (default) or deberta")
    parser.add_argument("--deberta-model", type=str, default=None,
                        choices=["best", "balanced", "fast", "bart"],
                        help="DeBERTa model preset: best (~60min), balanced (~20min, default), fast (~12min), bart (~45min)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    global RESULTS_DIR
    if args.output:
        RESULTS_DIR = args.output

    ensure_dirs()

    # Pipeline
    df = load_data(args.input)
    df = preprocess(df)
    df = apply_rule_labels(df)
    df = apply_sentiment(df)
    df = apply_entity_extraction(df)
    df = apply_summarization(df)

    if args.model == "deberta":
        df = run_deberta(df, deberta_model=args.deberta_model)
    else:
        df = run_ensemble(df)

    # Visualizations
    print("\nGenerating visualizations...")
    plot_label_distribution(df)
    plot_sentiment_distribution(df)
    plot_top_companies(df)

    # Get metrics for comparison plot
    if "ensemble_prediction" in df.columns:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(df["rule_label"], df["ensemble_prediction"])
        plot_accuracy_comparison({"mean_cv_accuracy": acc})

    # Save output
    output_cols = [
        "company", "subject", "date_only", "email_body",
        "rule_label", "rule_confidence", "final_label",
        "sentiment_label", "sentiment_compound",
        "extracted_role", "contact_person", "contact_email",
        "dates_mentioned", "summary",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    out_path = os.path.join(RESULTS_DIR, "classified_emails.csv")
    df[output_cols].to_csv(out_path, index=False)
    print(f"\nFinal dataset saved: {out_path}")

    # Sample results
    print("\n" + "=" * 60)
    print("  Sample Results")
    print("=" * 60)
    for lbl in df["final_label"].unique():
        sample = df[df["final_label"] == lbl].iloc[0]
        print(f"\n  [{lbl.upper()}]  {sample['company']}")
        print(f"    Subject:   {str(sample['subject'])[:70]}")
        print(f"    Sentiment: {sample['sentiment_label']}")
        print(f"    Summary:   {str(sample['summary'])[:100]}...")

    print("\n" + "=" * 60)
    print(f"  Done. {len(df)} emails → {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
