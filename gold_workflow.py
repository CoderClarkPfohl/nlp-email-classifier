"""Gold-label template generation and evaluation workflows."""

import math
import os
from collections import Counter
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from analysis.error_analysis import export_error_analysis
from models.feature_engineering import EnhancedFeatureTransformer
from models.rule_labeler import label_email
from models.svm_classifier import train_and_evaluate
from utils.entity_extraction import extract_entities
from utils.preprocessing import clean_email_body, preprocess_for_model
from utils.review_routing import review_statuses_from_probas
from models.sentiment import compute_sentiment
from utils.summarizer import summarize_email


LABELS = [
    "acceptance",
    "rejection",
    "interview",
    "action_required",
    "in_process",
    "unrelated",
]

REQUIRED_GOLD_COLUMNS = ["email_body", "true_label"]


def validate_gold_labels(df: pd.DataFrame, allowed_labels: Iterable[str] = LABELS) -> None:
    """Validate gold evaluation input before model work starts."""
    missing = [c for c in REQUIRED_GOLD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Gold CSV is missing required column(s): {missing}")

    invalid = sorted(set(df["true_label"].dropna()) - set(allowed_labels))
    if invalid:
        raise ValueError(
            f"Gold CSV contains invalid true_label value(s): {invalid}. "
            f"Allowed labels: {list(allowed_labels)}"
        )

    if df["true_label"].isna().any() or (df["true_label"].astype(str).str.strip() == "").any():
        raise ValueError("Gold CSV contains blank true_label values")


def _apply_gold_base_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run deterministic enrichment needed by gold evaluation."""
    df = df.dropna(subset=["email_body"]).drop_duplicates(subset=["email_body"]).reset_index(drop=True)
    df["clean_body"] = df["email_body"].apply(clean_email_body)
    df["model_body"] = df["email_body"].apply(preprocess_for_model)

    rule_results = df.apply(
        lambda r: label_email(str(r.get("subject", "")), str(r.get("email_body", ""))),
        axis=1,
    )
    df["rule_label"] = [r[0] for r in rule_results]
    df["rule_confidence"] = [r[1] for r in rule_results]

    sentiments = df["clean_body"].apply(compute_sentiment)
    df["sentiment_label"] = [s["label"] for s in sentiments]
    df["sentiment_compound"] = [s["compound"] for s in sentiments]
    df["sentiment_positive"] = [s["positive"] for s in sentiments]
    df["sentiment_negative"] = [s["negative"] for s in sentiments]

    entities = df.apply(
        lambda r: extract_entities(str(r.get("email_body", "")), str(r.get("company", ""))),
        axis=1,
    )
    df["extracted_role"] = [e["job_role"] for e in entities]
    df["contact_person"] = [e["contact_person"] for e in entities]
    df["contact_email"] = [e["contact_email"] for e in entities]
    df["dates_mentioned"] = [
        "; ".join(e["dates_mentioned"]) if e["dates_mentioned"] else ""
        for e in entities
    ]
    df["summary"] = df["clean_body"].apply(lambda t: summarize_email(t, max_sentences=2))
    return df


def make_gold_template(
    input_path: str,
    output_path: str = "data/gold_label_review_template.csv",
    sample_size: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a balanced manual-labeling template with blank true_label."""
    df = pd.read_csv(input_path).dropna(subset=["email_body"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["email_body"]).reset_index(drop=True)
    df["source_row"] = df.index

    rule_results = df.apply(
        lambda r: label_email(str(r.get("subject", "")), str(r.get("email_body", ""))),
        axis=1,
    )
    df["rule_label"] = [r[0] for r in rule_results]
    df["rule_confidence"] = [r[1] for r in rule_results]

    labels = sorted(df["rule_label"].unique())
    per_class = max(1, math.ceil(sample_size / max(len(labels), 1)))
    sampled = []
    for label in labels:
        group = df[df["rule_label"] == label]
        sampled.append(group.sample(n=min(per_class, len(group)), random_state=random_state))
    template = pd.concat(sampled, ignore_index=True)
    if len(template) > sample_size:
        template = template.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    preferred = [
        "source_row",
        "true_label",
        "rule_label",
        "rule_confidence",
        "company",
        "subject",
        "email_body",
    ]
    template["true_label"] = ""
    columns = [c for c in preferred if c in template.columns]
    template = template[columns]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    template.to_csv(output_path, index=False)
    return template


def plot_gold_confusion_matrix(cm, labels: List[str], output_path: str) -> None:
    """Save a gold confusion matrix image."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Gold Evaluation Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_gold(
    gold_path: str,
    results_dir: str = "results",
    n_folds: int = 5,
    export_errors: bool = False,
) -> pd.DataFrame:
    """Evaluate rule labels and the ensemble against human gold labels."""
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(gold_path)
    validate_gold_labels(df)
    df = _apply_gold_base_pipeline(df)

    texts = df["clean_body"].tolist()
    labels = df["true_label"].tolist()
    _, _, cv_preds, cv_probas, metrics = train_and_evaluate(
        texts,
        labels,
        n_folds=n_folds,
        use_stemming=True,
        feature_df=df,
        feature_transformer_factory=EnhancedFeatureTransformer,
    )

    df["ensemble_prediction"] = cv_preds
    df["final_label"] = cv_preds
    df["ensemble_confidence"] = cv_probas.max(axis=1).round(4)
    sorted_probas = cv_probas.copy()
    sorted_probas.sort(axis=1)
    df["ensemble_top2_gap"] = (
        sorted_probas[:, -1] - sorted_probas[:, -2]
        if sorted_probas.shape[1] > 1
        else sorted_probas[:, -1]
    ).round(4)
    df["review_status"] = review_statuses_from_probas(cv_probas)

    label_set = metrics["labels"]
    rule_report = classification_report(
        df["true_label"], df["rule_label"], labels=label_set, zero_division=0
    )
    ensemble_report = metrics["report_text"]
    rule_cm = confusion_matrix(df["true_label"], df["rule_label"], labels=label_set)

    report_path = os.path.join(results_dir, "gold_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("GOLD EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {gold_path}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Class distribution: {dict(sorted(Counter(labels).items()))}\n\n")
        f.write("Rule Labeler vs Gold\n")
        f.write("-" * 60 + "\n")
        f.write(rule_report)
        f.write("\n\nEnsemble vs Gold (Stratified CV)\n")
        f.write("-" * 60 + "\n")
        f.write(ensemble_report)
        f.write(f"\nMean fold accuracy: {metrics['mean_cv_accuracy']:.4f}\n")
        f.write(f"Per-fold accuracy: {[round(a, 4) for a in metrics['fold_accuracies']]}\n")
        f.write("\nRule confusion matrix:\n")
        f.write(str(rule_cm))
        f.write("\n")

    plot_gold_confusion_matrix(
        metrics["confusion_matrix"],
        label_set,
        os.path.join(results_dir, "gold_confusion_matrix.png"),
    )

    output_cols = [
        "company",
        "subject",
        "email_body",
        "true_label",
        "rule_label",
        "rule_confidence",
        "ensemble_prediction",
        "ensemble_confidence",
        "ensemble_top2_gap",
        "review_status",
        "sentiment_label",
        "sentiment_compound",
        "extracted_role",
        "contact_person",
        "contact_email",
        "dates_mentioned",
        "summary",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    predictions_path = os.path.join(results_dir, "gold_predictions.csv")
    df[output_cols].to_csv(predictions_path, index=False)

    if export_errors:
        export_error_analysis(
            df,
            os.path.join(results_dir, "gold_errors.csv"),
            true_col="true_label",
            pred_col="ensemble_prediction",
        )

    return df
