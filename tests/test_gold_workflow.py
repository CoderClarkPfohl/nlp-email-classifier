import pandas as pd
import pytest

from analysis.error_analysis import export_error_analysis
from gold_workflow import evaluate_gold, make_gold_template, validate_gold_labels


def _gold_rows():
    rows = []
    for i in range(6):
        rows.append(
            {
                "company": "OfferCo",
                "subject": f"Offer details {i}",
                "email_body": (
                    "We are pleased to offer you the Software Engineer position. "
                    f"Welcome aboard and your start date is May {i + 1}."
                ),
                "true_label": "acceptance",
            }
        )
        rows.append(
            {
                "company": "RejectCo",
                "subject": f"Application update {i}",
                "email_body": (
                    "We regret to inform you that we are not moving forward "
                    f"with your application for requisition {i}."
                ),
                "true_label": "rejection",
            }
        )
    return rows


def test_validate_gold_labels_requires_true_label():
    df = pd.DataFrame({"email_body": ["hello"]})
    with pytest.raises(ValueError, match="true_label"):
        validate_gold_labels(df)


def test_validate_gold_labels_rejects_invalid_label():
    df = pd.DataFrame({"email_body": ["hello"], "true_label": ["maybe"]})
    with pytest.raises(ValueError, match="invalid true_label"):
        validate_gold_labels(df)


def test_make_gold_template_balances_and_blanks_labels(tmp_path):
    input_path = tmp_path / "emails.csv"
    output_path = tmp_path / "template.csv"
    pd.DataFrame(_gold_rows()).drop(columns=["true_label"]).to_csv(input_path, index=False)

    template = make_gold_template(
        str(input_path),
        output_path=str(output_path),
        sample_size=4,
        random_state=0,
    )

    assert output_path.exists()
    assert len(template) == 4
    assert "true_label" in template.columns
    assert template["true_label"].tolist() == ["", "", "", ""]
    assert set(template["rule_label"]) == {"acceptance", "rejection"}


def test_evaluate_gold_writes_outputs(tmp_path):
    gold_path = tmp_path / "gold_labels.csv"
    results_dir = tmp_path / "results"
    pd.DataFrame(_gold_rows()).to_csv(gold_path, index=False)

    df = evaluate_gold(
        str(gold_path),
        results_dir=str(results_dir),
        n_folds=2,
        export_errors=True,
    )

    assert len(df) == 12
    assert (results_dir / "gold_classification_report.txt").exists()
    assert (results_dir / "gold_confusion_matrix.png").exists()
    assert (results_dir / "gold_predictions.csv").exists()
    assert (results_dir / "gold_errors.csv").exists()
    assert "review_status" in df.columns


def test_export_error_analysis_only_writes_misclassifications(tmp_path):
    df = pd.DataFrame(
        {
            "company": ["A", "B"],
            "subject": ["ok", "bad"],
            "email_body": ["right", "wrong"],
            "true_label": ["acceptance", "rejection"],
            "ensemble_prediction": ["acceptance", "acceptance"],
            "ensemble_confidence": [0.9, 0.6],
            "ensemble_top2_gap": [0.4, 0.1],
        }
    )
    output_path = tmp_path / "errors.csv"
    errors = export_error_analysis(df, str(output_path))
    written = pd.read_csv(output_path)

    assert len(errors) == 1
    assert len(written) == 1
    assert written.loc[0, "true_label"] == "rejection"
