import sys

import pandas as pd

import main


def test_main_make_gold_template_cli(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "emails.csv"
    rows = [
        {
            "company": "OfferCo",
            "subject": "Offer",
            "email_body": "We are pleased to offer you the position. Welcome aboard.",
        },
        {
            "company": "RejectCo",
            "subject": "Application update",
            "email_body": "We regret to inform you that we are not moving forward.",
        },
    ]
    pd.DataFrame(rows).to_csv(input_path, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--make-gold-template",
            "--input",
            str(input_path),
            "--gold-sample-size",
            "2",
        ],
    )

    main.main()

    assert (tmp_path / "data" / "gold_label_review_template.csv").exists()
    assert "Gold review template saved" in capsys.readouterr().out


def test_main_gold_cli_dispatches_to_evaluator(monkeypatch, tmp_path):
    calls = {}

    def fake_evaluate_gold(gold_path, results_dir="results", export_errors=False):
        calls["gold_path"] = gold_path
        calls["results_dir"] = results_dir
        calls["export_errors"] = export_errors
        return pd.DataFrame({"email_body": ["x"], "true_label": ["in_process"]})

    import gold_workflow

    monkeypatch.setattr(gold_workflow, "evaluate_gold", fake_evaluate_gold)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--gold",
            str(tmp_path / "gold.csv"),
            "--output",
            str(tmp_path / "results"),
            "--export-errors",
        ],
    )

    main.main()

    assert calls == {
        "gold_path": str(tmp_path / "gold.csv"),
        "results_dir": str(tmp_path / "results"),
        "export_errors": True,
    }


def test_main_ablation_cli_dispatches_to_runner(monkeypatch, tmp_path):
    calls = {}

    def fake_run_ablation(gold_path, results_dir="results"):
        calls["gold_path"] = gold_path
        calls["results_dir"] = results_dir
        return pd.DataFrame({"experiment": ["tfidf_only"], "accuracy": [1.0]})

    import experiments.run_ablation

    monkeypatch.setattr(experiments.run_ablation, "run_ablation", fake_run_ablation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--ablation",
            "--gold",
            str(tmp_path / "gold.csv"),
            "--output",
            str(tmp_path / "results"),
        ],
    )

    main.main()

    assert calls == {
        "gold_path": str(tmp_path / "gold.csv"),
        "results_dir": str(tmp_path / "results"),
    }
