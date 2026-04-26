"""
Tests for app/server.py — Flask web UI for email classification.

Tests the upload flow, validation, pipeline execution, results page,
and file downloads using Flask's test client (no live server needed).
"""

import io
import json
import os
import shutil
import pytest
import pandas as pd

from app.server import create_app, UPLOAD_DIR, _build_summary


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def app():
    application = create_app()
    application.config["TESTING"] = True
    yield application
    # Clean up any test session dirs
    if UPLOAD_DIR.exists():
        for p in UPLOAD_DIR.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)


@pytest.fixture
def client(app):
    return app.test_client()


def _make_csv_bytes(rows: list[dict]) -> bytes:
    """Build a CSV file as bytes from a list of dicts."""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def small_csv():
    """30 emails (5 per class) — enough for CalibratedClassifierCV(cv=3)."""
    base = [
        ("A", "Offer letter", "acceptance",
         "We are pleased to offer you the position of {}. Welcome aboard! Your start date is March 1."),
        ("B", "Application update", "rejection",
         "We regret to inform you that we will not be moving forward with your application for {} at this time."),
        ("C", "Interview invitation", "interview",
         "We would like to invite you for an interview for the {} position. Please schedule your phone screen."),
        ("D", "Action required: assessment", "action_required",
         "Please complete the online assessment for the {} role. The HackerRank coding challenge must be done by Friday."),
        ("E", "Application received", "in_process",
         "Thank you for applying to the {} role. We have received your application and our team will review it carefully."),
        ("F", "New jobs matching your search", "unrelated",
         "Here are new {} jobs that match your profile. Unsubscribe from all alerts. Browse similar jobs on our platform."),
    ]
    roles = ["Engineer", "Analyst", "Manager", "Scientist", "Designer"]
    rows = []
    for company, subject, _label, body_tmpl in base:
        for role in roles:
            rows.append({
                "company": company,
                "subject": subject,
                "email_body": body_tmpl.format(role),
            })
    return _make_csv_bytes(rows)


# ─────────────────────────────────────────────────────────────
#  Index page
# ─────────────────────────────────────────────────────────────

class TestIndex:
    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_has_upload_form(self, client):
        resp = client.get("/")
        assert b"email_body" in resp.data
        assert b"Classify" in resp.data

    def test_index_shows_categories(self, client):
        resp = client.get("/")
        for cat in [b"acceptance", b"rejection", b"interview",
                    b"action_required", b"in_process", b"unrelated"]:
            assert cat in resp.data


# ─────────────────────────────────────────────────────────────
#  Upload validation
# ─────────────────────────────────────────────────────────────

class TestUploadValidation:
    def test_no_file_flashes_error(self, client):
        resp = client.post("/classify", data={}, follow_redirects=True)
        assert b"No file uploaded" in resp.data

    def test_empty_filename_flashes_error(self, client):
        data = {"file": (io.BytesIO(b""), "")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert b"Please upload a .csv file" in resp.data

    def test_non_csv_extension_rejected(self, client):
        data = {"file": (io.BytesIO(b"data"), "test.txt")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert b"Please upload a .csv file" in resp.data

    def test_missing_email_body_column(self, client):
        bad_csv = b"name,age\nAlice,30\n"
        data = {"file": (io.BytesIO(bad_csv), "bad.csv")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert b"email_body" in resp.data

    def test_empty_bodies_rejected(self, client):
        csv_data = b"email_body\n\n\n"
        data = {"file": (io.BytesIO(csv_data), "empty.csv")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert b"no non-empty" in resp.data


# ─────────────────────────────────────────────────────────────
#  Full pipeline via upload
# ─────────────────────────────────────────────────────────────

class TestClassifyFlow:
    def test_classify_redirects_to_results(self, client, small_csv):
        data = {"file": (io.BytesIO(small_csv), "test.csv")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data")
        assert resp.status_code == 302
        assert "/results/" in resp.headers["Location"]

    def test_results_page_renders(self, client, small_csv):
        data = {"file": (io.BytesIO(small_csv), "test.csv")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert resp.status_code == 200
        assert b"Classification Results" in resp.data
        assert b"30" in resp.data  # total emails
        assert b"Download Excel" in resp.data
        assert b"Download CSV" in resp.data

    def test_results_page_shows_preview_rows(self, client, small_csv):
        data = {"file": (io.BytesIO(small_csv), "test.csv")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        # Should see some subject text from our fixture
        assert b"Offer letter" in resp.data or b"Application" in resp.data

    def test_results_page_has_charts(self, client, small_csv):
        data = {"file": (io.BytesIO(small_csv), "test.csv")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert b"catChart" in resp.data
        assert b"sentChart" in resp.data

    def _get_session_id(self, client, small_csv):
        """Helper: upload and return the session ID."""
        data = {"file": (io.BytesIO(small_csv), "test.csv")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data")
        return resp.headers["Location"].split("/results/")[-1]

    def test_download_xlsx(self, client, small_csv):
        sid = self._get_session_id(client, small_csv)
        resp = client.get(f"/download/{sid}/xlsx")
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.content_type or "octet-stream" in resp.content_type

    def test_download_csv(self, client, small_csv):
        sid = self._get_session_id(client, small_csv)
        resp = client.get(f"/download/{sid}/csv")
        assert resp.status_code == 200
        # Parse the downloaded CSV and check structure
        csv_text = resp.data.decode("utf-8")
        assert "final_label" in csv_text
        assert "email_body" in csv_text

    def test_download_invalid_type(self, client, small_csv):
        sid = self._get_session_id(client, small_csv)
        resp = client.get(f"/download/{sid}/pdf")
        # Should redirect to index
        assert resp.status_code == 302
        resp2 = client.get(f"/download/{sid}/pdf", follow_redirects=True)
        assert b"Invalid file type" in resp2.data

    def test_api_summary_endpoint(self, client, small_csv):
        sid = self._get_session_id(client, small_csv)
        resp = client.get(f"/api/summary/{sid}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 30
        assert "categories" in data
        assert "sentiment" in data

    def test_summary_uses_true_label_accuracy_when_available(self):
        df = pd.DataFrame(
            {
                "final_label": ["acceptance", "rejection", "interview"],
                "true_label": ["acceptance", "in_process", "interview"],
                "ensemble_confidence": [0.9, 0.8, 0.7],
                "sentiment_label": ["positive", "neutral", "positive"],
            }
        )
        summary = _build_summary(df, {"mean_cv_accuracy": 1.0})
        assert summary["mean_accuracy"] == 66.7
        assert summary["accuracy_label"] == "Accuracy vs True Label"
        assert summary["has_true_labels"] is True


# ─────────────────────────────────────────────────────────────
#  Edge cases
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_expired_session_redirects(self, client):
        resp = client.get("/results/nonexistent123", follow_redirects=True)
        assert b"expired" in resp.data or b"not found" in resp.data

    def test_expired_download_redirects(self, client):
        resp = client.get("/download/nonexistent123/xlsx", follow_redirects=True)
        assert b"not found" in resp.data or b"expired" in resp.data

    def test_api_summary_404_for_unknown(self, client):
        resp = client.get("/api/summary/nonexistent123")
        assert resp.status_code == 404

    def test_duplicate_emails_deduplicated(self, client, small_csv):
        """Pipeline should remove duplicate email bodies before classifying."""
        # Take the 30-email fixture and add 10 duplicates (rows 0-9 again)
        import csv as csv_mod
        lines = small_csv.decode("utf-8").splitlines()
        reader = list(csv_mod.DictReader(lines))
        all_rows = reader + reader[:10]  # 40 rows, 30 unique
        csv_bytes = _make_csv_bytes(all_rows)
        data = {"file": (io.BytesIO(csv_bytes), "dupes.csv")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert resp.status_code == 200
        assert b"Classification Results" in resp.data
        # Should report 30 (not 40)
        assert b"30" in resp.data

    def test_csv_with_only_email_body(self, client, small_csv):
        """Minimal CSV with just email_body works (no company/subject columns)."""
        # Rebuild small_csv keeping only email_body
        import csv as csv_mod
        lines = small_csv.decode("utf-8").splitlines()
        reader = csv_mod.DictReader(lines)
        minimal_rows = [{"email_body": row["email_body"]} for row in reader]
        csv_bytes = _make_csv_bytes(minimal_rows)
        data = {"file": (io.BytesIO(csv_bytes), "minimal.csv")}
        resp = client.post("/classify", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert resp.status_code == 200
        assert b"Classification Results" in resp.data
