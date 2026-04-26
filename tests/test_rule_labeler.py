"""Tests for models/rule_labeler.py."""

import pytest
from models.rule_labeler import label_email


class TestRejection:
    def test_strong_rejection_signal(self):
        label, conf = label_email("Application update", "We regret to inform you that we have decided not to move forward.")
        assert label == "rejection"
        assert conf >= 0.85

    def test_unfortunately_keyword(self):
        label, _ = label_email("Update", "Unfortunately we will not be proceeding with your application.")
        assert label == "rejection"

    def test_position_filled(self):
        label, _ = label_email("Status", "The position has been filled by another candidate.")
        assert label == "rejection"


class TestAcceptance:
    def test_offer_letter(self):
        label, conf = label_email("Congratulations!", "We are pleased to offer you the position. Your start date is January 15.")
        assert label == "acceptance"
        assert conf >= 0.85

    def test_welcome_aboard(self):
        label, _ = label_email("Welcome!", "Welcome aboard! Your onboarding begins next week.")
        assert label == "acceptance"


class TestInterview:
    def test_interview_invitation(self):
        label, conf = label_email("Interview Invitation", "We'd like to invite you for an interview next week.")
        assert label == "interview"
        assert conf >= 0.70

    def test_phone_screen(self):
        label, _ = label_email("Phone screen", "We'd like to schedule a phone screen with you.")
        assert label == "interview"

    def test_schedule_interview(self):
        label, _ = label_email("Next steps", "We'd like to schedule an interview at your convenience.")
        assert label == "interview"


class TestActionRequired:
    def test_assessment(self):
        label, conf = label_email("Action Required", "Please complete your assessment within 5 days.")
        assert label == "action_required"
        assert conf >= 0.70

    def test_coding_challenge(self):
        label, _ = label_email("Next steps", "We'd like you to complete a coding challenge on HackerRank.")
        assert label == "action_required"

    def test_background_check(self):
        label, _ = label_email("Background check", "Please complete the background check authorization.")
        assert label == "action_required"


class TestInProcess:
    def test_application_received(self):
        label, _ = label_email("Thank you", "We have received your application and will review your qualifications.")
        assert label == "in_process"

    def test_under_review(self):
        label, _ = label_email("Status", "Your application is currently under review by our team.")
        assert label == "in_process"

    def test_thanks_for_applying(self):
        label, _ = label_email("Thanks!", "Thank you for applying to our Data Analyst position.")
        assert label == "in_process"


class TestUnrelated:
    def test_job_alert(self):
        label, _ = label_email("Job alerts", "New jobs for you this week. Unsubscribe from all alerts.")
        assert label == "unrelated"

    def test_newsletter(self):
        label, _ = label_email("Newsletter", "Check out our newsletter and marketing preferences.")
        assert label == "unrelated"

    def test_password_reset(self):
        label, _ = label_email("Password reset", "You requested a password reset. Click here to reset.")
        assert label == "unrelated"


class TestConfidence:
    def test_no_signals_defaults_to_in_process(self):
        # Text must mention "application" or similar to avoid the unrelated heuristic
        label, conf = label_email("", "Thank you for your application. We will get back to you.")
        assert label == "in_process"

    def test_strong_signal_high_confidence(self):
        _, conf = label_email("Offer", "We are pleased to offer you the position. Welcome aboard! Your start date is set.")
        assert conf >= 0.85

    def test_returns_tuple(self):
        result = label_email("Test", "Test body")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)


class TestMultiSignal:
    def test_strongest_signal_wins(self):
        # Has both in_process AND action_required signals; action should win
        label, _ = label_email(
            "Action Required",
            "Thank you for applying. Please complete your assessment. Take the online assessment on HackerRank."
        )
        assert label == "action_required"
