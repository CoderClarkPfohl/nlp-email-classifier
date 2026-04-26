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

    def test_pursue_other_applicants(self):
        label, conf = label_email(
            "Follow-up regarding role",
            "The team has chosen to pursue other applicants for the position.",
        )
        assert label == "rejection"
        assert conf >= 0.70

    def test_declining_to_proceed(self):
        label, conf = label_email(
            "Status update for analyst",
            "At this time we are declining to proceed to the next stage.",
        )
        assert label == "rejection"
        assert conf >= 0.70

    def test_not_continue_with_candidacy(self):
        label, conf = label_email(
            "Application outcome",
            "Unfortunately, we will not continue with your candidacy for this requisition.",
        )
        assert label == "rejection"
        assert conf >= 0.70

    def test_not_advance_application(self):
        label, conf = label_email(
            "Application update",
            "After comparing all candidates, we decided not to advance your application.",
        )
        assert label == "rejection"
        assert conf >= 0.70

    def test_selected_another_applicant(self):
        label, conf = label_email(
            "Application decision",
            "The team selected another applicant for the next stage, so we are closing this application.",
        )
        assert label == "rejection"
        assert conf >= 0.70


class TestAcceptance:
    def test_offer_letter(self):
        label, conf = label_email("Congratulations!", "We are pleased to offer you the position. Your start date is January 15.")
        assert label == "acceptance"
        assert conf >= 0.85

    def test_welcome_aboard(self):
        label, _ = label_email("Welcome!", "Welcome aboard! Your onboarding begins next week.")
        assert label == "acceptance"

    def test_employment_offer(self):
        label, conf = label_email(
            "Company employment offer",
            "We approved an employment offer and are excited to have you join as an analyst.",
        )
        assert label == "acceptance"
        assert conf >= 0.70

    def test_offer_package(self):
        label, conf = label_email(
            "Next step: review your offer package",
            "We are extending a job offer and the written offer package explains compensation.",
        )
        assert label == "acceptance"
        assert conf >= 0.70

    def test_hiring_package_offer_stage(self):
        label, conf = label_email(
            "Your hiring package",
            "We are excited to move from interviews to offer stage and have prepared a hiring package.",
        )
        assert label == "acceptance"
        assert conf >= 0.70

    def test_join_us_attached_terms(self):
        label, conf = label_email(
            "Next steps to join company",
            "Good news from the hiring team. We would like you to join us. Please review the attached terms and proposed start timeline.",
        )
        assert label == "acceptance"
        assert conf >= 0.70


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

    def test_first_conversation_request(self):
        label, conf = label_email(
            "Conversation request",
            "We would like to arrange a first conversation with the hiring team. Please send availability for a 30 minute video call.",
        )
        assert label == "interview"
        assert conf >= 0.70


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

    def test_missing_profile_information(self):
        label, conf = label_email(
            "Missing information for your application",
            "Your candidate profile is missing required information. Please upload your updated resume and complete the work authorization questions.",
        )
        assert label == "action_required"
        assert conf >= 0.70

    def test_screening_questionnaire_pending(self):
        label, conf = label_email(
            "Candidate task pending",
            "There is a pending task on your candidate profile. Please answer the screening questionnaire. Your application will remain paused until the task is complete.",
        )
        assert label == "action_required"
        assert conf >= 0.70

    def test_required_questionnaire_needs_attention(self):
        label, conf = label_email(
            "Action required for application",
            "Your application needs attention. Please complete the required questionnaire before review can continue.",
        )
        assert label == "action_required"
        assert conf >= 0.70

    def test_missing_required_item_requested_form(self):
        label, conf = label_email(
            "More information needed",
            "Your application is missing a required item. Please complete the requested form to keep the process active.",
        )
        assert label == "action_required"
        assert conf >= 0.70


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

    def test_profile_being_reviewed(self):
        label, conf = label_email(
            "Your profile is being reviewed",
            "Your materials are in our recruiting system. If the team sees a possible fit, someone will contact you about next steps. Thank you for considering us.",
        )
        assert label == "in_process"
        assert conf >= 0.70

    def test_we_have_your_materials(self):
        label, conf = label_email(
            "We have your materials",
            "The hiring group is reviewing profiles over the next few weeks. You do not need to reply to this automated confirmation.",
        )
        assert label == "in_process"
        assert conf >= 0.70


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

    def test_virtual_event_not_application_update(self):
        label, conf = label_email(
            "Upcoming virtual recruiting event",
            "Registration is open to anyone interested in learning about the industry. Attending the event does not create or change a job application.",
        )
        assert label == "unrelated"
        assert conf >= 0.70


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
