#!/usr/bin/env python3
"""
Flask web UI for the NLP Email Classifier.

Upload a CSV → full pipeline runs → view results + download colored Excel.

Usage:
    python -m app.server            # from project root
    python app/server.py            # also works (adds parent to sys.path)
"""

import os
import sys
import uuid
import shutil
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path so imports work when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, flash, jsonify,
)

from utils.preprocessing import clean_email_body, preprocess_for_model
from utils.entity_extraction import extract_entities
from utils.summarizer import summarize_email
from utils.excel_export import export_to_excel, CATEGORY_COLORS
from models.rule_labeler import label_email
from models.sentiment import compute_sentiment
from models.svm_classifier import train_and_evaluate
from models.feature_engineering import EnhancedFeatureTransformer
from utils.review_routing import review_statuses_from_probas

import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────

UPLOAD_DIR = Path(_PROJECT_ROOT) / "app" / "sessions"
ALLOWED_EXTENSIONS = {"csv"}
SESSION_TTL_SECONDS = 3600  # 1 hour

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET", "nlp-classifier-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────
#  Session cleanup (background thread)
# ─────────────────────────────────────────────────────────────

def _cleanup_loop():
    """Delete session folders older than SESSION_TTL_SECONDS."""
    while True:
        time.sleep(300)
        if not UPLOAD_DIR.exists():
            continue
        now = time.time()
        for p in UPLOAD_DIR.iterdir():
            if p.is_dir() and (now - p.stat().st_mtime) > SESSION_TTL_SECONDS:
                shutil.rmtree(p, ignore_errors=True)

_cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)


# ─────────────────────────────────────────────────────────────
#  Pipeline (reuses existing project functions)
# ─────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame) -> tuple:
    """
    Run the full NLP pipeline on a DataFrame.
    Returns (df_enriched, metrics_dict).
    """
    # Preprocess
    df["clean_body"] = df["email_body"].apply(clean_email_body)
    df["model_body"] = df["email_body"].apply(preprocess_for_model)

    # Rule labels
    results = df.apply(
        lambda r: label_email(str(r.get("subject", "")),
                              str(r.get("email_body", ""))),
        axis=1,
    )
    df["rule_label"] = [r[0] for r in results]
    df["rule_confidence"] = [r[1] for r in results]

    # Sentiment
    sents = df["clean_body"].apply(compute_sentiment)
    df["sentiment_label"] = [s["label"] for s in sents]
    df["sentiment_compound"] = [s["compound"] for s in sents]

    # Entity extraction
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

    # Summarization
    df["summary"] = df["clean_body"].apply(
        lambda t: summarize_email(t, max_sentences=2)
    )

    # Ensemble classifier with fold-local enhanced features + stemming
    tfidf, ensemble, cv_preds, cv_probas, metrics = train_and_evaluate(
        df["clean_body"].tolist(), df["rule_label"].tolist(), n_folds=5,
        use_stemming=True,
        feature_df=df,
        feature_transformer_factory=EnhancedFeatureTransformer,
    )
    df["final_label"] = cv_preds
    df["ensemble_confidence"] = cv_probas.max(axis=1).round(4)
    sorted_p = np.sort(cv_probas, axis=1)
    df["ensemble_top2_gap"] = (sorted_p[:, -1] - sorted_p[:, -2]).round(4)
    df["review_status"] = review_statuses_from_probas(cv_probas)
    if "true_label" in df.columns:
        true_labels = df["true_label"].fillna("").astype(str).str.strip()
        df["model_matches_true"] = df["final_label"].astype(str) == true_labels

    return df, metrics


def _build_summary(df: pd.DataFrame, metrics: dict) -> dict:
    """Build a JSON-serializable summary dict for the results page."""
    label_counts = df["final_label"].value_counts().to_dict()
    total = len(df)
    categories = []
    for cat in sorted(label_counts.keys()):
        cnt = label_counts[cat]
        colors = CATEGORY_COLORS.get(cat, {"fill": "FFFFFF", "font": "000000"})
        categories.append({
            "name": cat,
            "count": cnt,
            "pct": round(cnt / total * 100, 1),
            "fill": f"#{colors['fill']}",
            "font": f"#{colors['font']}",
        })

    sentiment_counts = df["sentiment_label"].value_counts().to_dict()
    has_true_labels = (
        "true_label" in df.columns
        and df["true_label"].notna().any()
        and (df["true_label"].astype(str).str.strip() != "").any()
    )
    has_true_labels = bool(has_true_labels)
    if has_true_labels:
        true_labels = df["true_label"].fillna("").astype(str).str.strip()
        mean_accuracy = round((df["final_label"].astype(str) == true_labels).mean() * 100, 1)
        accuracy_label = "Accuracy vs True Label"
    else:
        mean_accuracy = round(metrics["mean_cv_accuracy"] * 100, 1)
        accuracy_label = "Pseudo-label CV Agreement"

    # Sentiment breakdown per category
    cross = {}
    for cat in sorted(label_counts.keys()):
        subset = df[df["final_label"] == cat]
        cross[cat] = subset["sentiment_label"].value_counts().to_dict()

    return {
        "total": total,
        "mean_accuracy": mean_accuracy,
        "accuracy_label": accuracy_label,
        "has_true_labels": has_true_labels,
        "mean_confidence": round(df["ensemble_confidence"].mean() * 100, 1),
        "categories": categories,
        "true_categories": (
            df["true_label"].value_counts().to_dict() if has_true_labels else {}
        ),
        "sentiment": sentiment_counts,
        "sentiment_cross": cross,
    }


# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "" or not _allowed_file(file.filename):
        flash("Please upload a .csv file.", "error")
        return redirect(url_for("index"))

    # Create session directory
    session_id = uuid.uuid4().hex[:12]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save upload
    upload_path = session_dir / "upload.csv"
    file.save(str(upload_path))

    # Load and validate
    try:
        df = pd.read_csv(upload_path)
    except Exception as e:
        flash(f"Could not parse CSV: {e}", "error")
        shutil.rmtree(session_dir, ignore_errors=True)
        return redirect(url_for("index"))

    if "email_body" not in df.columns:
        flash(
            f"CSV must have an 'email_body' column. Found: {list(df.columns)}",
            "error",
        )
        shutil.rmtree(session_dir, ignore_errors=True)
        return redirect(url_for("index"))

    df = df.dropna(subset=["email_body"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["email_body"]).reset_index(drop=True)

    if len(df) == 0:
        flash("CSV has no non-empty email bodies.", "error")
        shutil.rmtree(session_dir, ignore_errors=True)
        return redirect(url_for("index"))

    # Run pipeline
    try:
        df, metrics = run_pipeline(df)
    except Exception as e:
        flash(f"Pipeline error: {e}", "error")
        shutil.rmtree(session_dir, ignore_errors=True)
        return redirect(url_for("index"))

    # Save outputs
    output_cols = [
        "company", "subject", "date_only", "email_body",
        "true_label", "rule_label", "rule_confidence", "final_label", "model_matches_true",
        "ensemble_confidence", "ensemble_top2_gap", "review_status",
        "sentiment_label", "sentiment_compound",
        "extracted_role", "contact_person", "contact_email",
        "dates_mentioned", "summary",
    ]
    output_cols = [c for c in output_cols if c in df.columns]

    csv_out = session_dir / "classified_emails.csv"
    xlsx_out = session_dir / "classified_emails.xlsx"
    df[output_cols].to_csv(str(csv_out), index=False)
    export_to_excel(df[output_cols], str(xlsx_out), label_column="final_label")

    # Save summary JSON
    import json
    summary = _build_summary(df, metrics)
    with open(session_dir / "summary.json", "w") as f:
        json.dump(summary, f)

    # Save preview rows (first 50) for the results page
    preview_cols = ["company", "subject", "true_label", "final_label",
                    "model_matches_true", "ensemble_confidence", "review_status",
                    "sentiment_label", "summary"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    preview = df[preview_cols].head(50).to_dict(orient="records")
    with open(session_dir / "preview.json", "w") as f:
        json.dump(preview, f)

    return redirect(url_for("results", session_id=session_id))


@app.route("/results/<session_id>")
def results(session_id):
    import json
    session_dir = UPLOAD_DIR / session_id

    summary_path = session_dir / "summary.json"
    preview_path = session_dir / "preview.json"
    if not summary_path.exists():
        flash("Session expired or not found.", "error")
        return redirect(url_for("index"))

    with open(summary_path) as f:
        summary = json.load(f)
    with open(preview_path) as f:
        preview = json.load(f)

    return render_template(
        "results.html",
        session_id=session_id,
        summary=summary,
        preview=preview,
        category_colors=CATEGORY_COLORS,
    )


@app.route("/download/<session_id>/<filetype>")
def download(session_id, filetype):
    session_dir = UPLOAD_DIR / session_id

    if filetype not in ("xlsx", "csv"):
        flash("Invalid file type.", "error")
        return redirect(url_for("index"))

    if filetype == "xlsx":
        path = session_dir / "classified_emails.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        path = session_dir / "classified_emails.csv"
        mime = "text/csv"

    if not path.exists():
        flash("File not found — session may have expired.", "error")
        return redirect(url_for("index"))

    return send_file(str(path), mimetype=mime, as_attachment=True,
                     download_name=path.name)


@app.route("/api/summary/<session_id>")
def api_summary(session_id):
    """JSON endpoint for chart data."""
    import json
    path = UPLOAD_DIR / session_id / "summary.json"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def create_app():
    """Factory for test clients and WSGI servers."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return app


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_thread.start()
    print(f"\n  NLP Email Classifier — Web UI")
    print(f"  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)
