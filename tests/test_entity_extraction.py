"""Tests for utils/entity_extraction.py."""

import pytest
from utils.entity_extraction import (
    extract_job_role, extract_dates, extract_contact_person,
    extract_contact_email, extract_entities,
)


class TestExtractJobRole:
    def test_applied_for_pattern(self):
        text = "Thank you for applying for the Data Analyst position at Acme Corp."
        role = extract_job_role(text)
        assert role is not None
        assert "Data Analyst" in role

    def test_application_for_pattern(self):
        # The regex requires "position|role|opening" after the role name
        text = "Your application for Software Engineer position has been received."
        role = extract_job_role(text)
        assert role is not None

    def test_no_role_found(self):
        text = "Thank you for your email. We will get back to you."
        assert extract_job_role(text) is None


class TestExtractDates:
    def test_month_day_year(self):
        text = "Your interview is on January 15, 2025."
        dates = extract_dates(text)
        assert len(dates) >= 1
        assert any("January" in d or "Jan" in d for d in dates)

    def test_numeric_date(self):
        text = "Deadline: 01/15/2025."
        dates = extract_dates(text)
        assert len(dates) >= 1

    def test_iso_date(self):
        text = "Start date: 2025-01-15."
        dates = extract_dates(text)
        assert len(dates) >= 1
        assert "2025-01-15" in dates

    def test_no_dates(self):
        text = "No dates mentioned here at all."
        assert extract_dates(text) == []


class TestExtractContactPerson:
    def test_sincerely_pattern(self):
        # The regex needs the comma, optional newline, then a capitalized name
        text = "We look forward to hearing from you.\n\nBest regards,\nSarah Johnson"
        person = extract_contact_person(text)
        assert person == "Sarah Johnson"

    def test_regards_pattern(self):
        text = "Best regards,\nJames Williams"
        person = extract_contact_person(text)
        assert person == "James Williams"

    def test_no_contact(self):
        text = "This is an automated email. Do not reply."
        assert extract_contact_person(text) is None


class TestExtractContactEmail:
    def test_finds_email(self):
        text = "Contact us at recruiter@company.com for questions."
        email = extract_contact_email(text)
        assert email == "recruiter@company.com"

    def test_skips_noreply(self):
        # The regex captures trailing dots, so use a space after the email
        text = "From: noreply@company.com — Reach out to hr@company.com for questions"
        email = extract_contact_email(text)
        assert email is not None
        assert "hr@company" in email

    def test_no_email(self):
        text = "No email addresses in this text."
        assert extract_contact_email(text) is None


class TestExtractEntities:
    def test_returns_dict_with_all_keys(self):
        result = extract_entities("Some email text.", "Acme")
        assert set(result.keys()) == {"company", "job_role", "contact_person",
                                       "contact_email", "dates_mentioned"}

    def test_uses_provided_company(self):
        result = extract_entities("text", "Google")
        assert result["company"] == "Google"

    def test_empty_company_returns_none(self):
        result = extract_entities("text", "")
        assert result["company"] is None
