"""Ablation study runner for gold-labeled email classification."""

import os

import pandas as pd

from gold_workflow import _apply_gold_base_pipeline, validate_gold_labels
from models.feature_engineering import EnhancedFeatureTransformer
from models.svm_classifier import train_and_evaluate


def run_ablation(
    gold_path: str,
    results_dir: str = "results",
    n_folds: int = 5,
) -> pd.DataFrame:
    """Compare feature groups on a gold-labeled dataset."""
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(gold_path)
    validate_gold_labels(df)
    df = _apply_gold_base_pipeline(df)

    texts = df["clean_body"].tolist()
    labels = df["true_label"].tolist()
    configs = [
        ("tfidf_only", None),
        (
            "tfidf_keywords",
            lambda: EnhancedFeatureTransformer(
                include_lm=False,
                include_centroids=False,
                include_keywords=True,
                include_ner=False,
                include_sentiment=False,
            ),
        ),
        (
            "tfidf_keywords_ner_sentiment",
            lambda: EnhancedFeatureTransformer(
                include_lm=False,
                include_centroids=False,
                include_keywords=True,
                include_ner=True,
                include_sentiment=True,
            ),
        ),
        ("tfidf_full_enhanced", EnhancedFeatureTransformer),
    ]

    rows = []
    for name, factory in configs:
        _, _, _, _, metrics = train_and_evaluate(
            texts,
            labels,
            n_folds=n_folds,
            use_stemming=True,
            feature_df=df if factory is not None else None,
            feature_transformer_factory=factory,
        )
        report = metrics["report_dict"]
        rows.append(
            {
                "experiment": name,
                "accuracy": report["accuracy"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "mean_fold_accuracy": metrics["mean_cv_accuracy"],
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(results_dir, "ablation_results.csv"), index=False)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run gold-label feature ablations")
    parser.add_argument("--gold", required=True, help="Path to gold_labels.csv")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--folds", type=int, default=5, help="Requested CV folds")
    args = parser.parse_args()

    results = run_ablation(args.gold, results_dir=args.output, n_folds=args.folds)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
