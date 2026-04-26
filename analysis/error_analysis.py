"""Gold-label error analysis exports."""

import os

import pandas as pd


def export_error_analysis(
    df: pd.DataFrame,
    output_path: str,
    true_col: str = "true_label",
    pred_col: str = "ensemble_prediction",
) -> pd.DataFrame:
    """Write misclassified gold examples and return the exported DataFrame."""
    if true_col not in df.columns:
        raise ValueError(f"Missing required column: {true_col}")
    if pred_col not in df.columns:
        raise ValueError(f"Missing required column: {pred_col}")

    errors = df[df[true_col] != df[pred_col]].copy()
    if "email_body" in errors.columns:
        errors["email_preview"] = (
            errors["email_body"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str[:240]
        )

    preferred = [
        "company",
        "subject",
        "true_label",
        pred_col,
        "rule_label",
        "rule_confidence",
        "ensemble_confidence",
        "ensemble_top2_gap",
        "review_status",
        "email_preview",
    ]
    cols = [c for c in preferred if c in errors.columns]
    export_df = errors[cols] if cols else errors

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    export_df.to_csv(output_path, index=False)
    return export_df
