"""
Rule-based email labeler — v2 (improved).
Generates pseudo-labels for job application emails based on keyword/phrase
matching heuristics. These labels serve as training data for ML classifiers.

Key improvements over v1:
- Much broader phrase lists for minority classes (action_required, interview)
- Multi-signal scoring (when email matches multiple categories, strongest wins)
- Subject-line weighting (subjects are more reliable signals)
- Confidence scoring based on margin between top two categories

Labels
------
- acceptance      : offer letter, congratulations on being hired
- rejection       : application declined
- interview       : interview invitation or scheduling
- action_required : assessment, coding challenge, task to complete
- in_process      : application received, under review
- unrelated       : newsletters, marketing, account setup emails
"""

from typing import Tuple, Dict


def _count_matches(text: str, phrases: list) -> int:
    """Count how many phrases appear in the text."""
    return sum(1 for p in phrases if p in text)


def _has_any(text: str, phrases: list) -> bool:
    return any(p in text for p in phrases)


# ── Phrase dictionaries ──

REJECTION_STRONG = [
    "regret to inform", "not moving forward", "decided not to proceed",
    "not be moving forward", "will not be advancing", "not been selected",
    "unable to offer", "decided to move ahead with other candidates",
    "decided not to move forward", "moving forward with other candidates",
    "we're sorry to let you know", "position has been filled",
    "we have decided to move forward with other", "not a match for",
    "not be continuing", "will not be progressing",
    "not proceeding with your", "have chosen not to",
    "no longer under consideration", "not advance your candidacy",
    "not advance your application", "decided not to advance your application",
    "pursue other candidates", "pursue other applicants",
    "chosen to pursue other applicants", "move on with other applicants",
    "selected candidates whose experience",
    "unable to offer you this position",
    "not be moving ahead", "will not be moving ahead",
    "declining to proceed", "declining to proceed to the next stage",
    "not selected for the next round",
    "not continue with your candidacy",
    "not selected for further consideration",
    "closing your current application",
    "will not continue with your candidacy",
    "decided to continue with candidates",
    "continue with candidates whose background",
    "selected another applicant",
    "not able to proceed with your candidacy",
    "not able to proceed",
    "closing this application",
    "will not move forward",
]

REJECTION_MEDIUM = [
    "unfortunately", "we will not", "cannot offer",
    "no longer being considered", "not selected for",
    "other candidates whom we feel", "we regret",
    "we're unable to", "not the right fit",
]

ACCEPTANCE_PHRASES = [
    "pleased to offer", "congratulations on your offer",
    "we are excited to extend", "offer of employment",
    "welcome aboard", "offer letter", "accepted your offer",
    "start date is", "your start date", "onboarding",
    "compensation package", "pleased to extend",
    "thrilled to welcome you", "welcome to the team",
    "employment offer", "extend an offer",
    "would like to extend an offer",
    "approved an employment offer",
    "excited to have you join",
    "have you join",
    "welcoming you to the team",
    "delighted to offer", "written offer",
    "offer package", "extending a job offer",
    "formal offer", "approved your formal offer",
    "selected to join",
    "ready to make you an offer",
    "hiring package",
    "offer stage",
    "happy to send your offer letter",
    "looking forward to having you aboard",
    "employment agreement",
    "move ahead with bringing you",
    "bringing you onto",
    "would like you to join",
    "attached terms",
    "proposed start timeline",
    "welcome you as our next",
    "join us",
]

INTERVIEW_STRONG = [
    "schedule an interview", "interview invitation",
    "like to invite you", "interview has been scheduled",
    "phone screen", "invite you for an interview",
    "book your interview", "schedule your interview",
    "video interview", "on-site interview",
    "virtual interview", "panel interview",
    "interview with our", "technical interview",
    "behavioral interview", "interview slot",
    "calendar invite for", "interview on",
    "we'd like to speak with you", "introductory call",
    "screening call", "initial call with",
    "would like to set up a call",
    "like to discuss your application",
    "recruiter will be reaching out",
    "would like to arrange a first conversation",
    "would like to arrange a recruiter phone screen",
    "would like to meet with you",
    "speak with the hiring team",
    "conversation request",
]

INTERVIEW_MEDIUM = [
    "next round", "we'd like to learn more about you",
    "meet with our team", "hiring manager would like",
    "move you to the next step", "advance to the next stage",
    "progressed to the next", "selected for the initial",
    "shortlisted", "we were impressed",
    "like to move forward with your candidacy",
    "first conversation", "conversation with",
    "calendar link to choose a time",
    "send availability",
]

ACTION_STRONG = [
    "complete your assessment", "action required",
    "coding challenge", "take the assessment",
    "complete the assessment", "online assessment",
    "technical assessment", "hackerrank", "codility",
    "codesignal", "hirevue", "karat",
    "assessment invitation", "please complete the following",
    "finish your application", "additional information needed",
    "please complete", "take our assessment",
    "complete this assessment", "invited to take",
    "complete the following steps", "task for you to complete",
    "pre-employment", "background check",
    "missing required information",
    "candidate profile is missing",
    "upload your updated resume",
    "work authorization questions",
    "cannot be reviewed further",
    "requires a short questionnaire",
    "finish candidate task",
    "submit the form",
    "need additional documents",
    "requested documents",
    "upload your current resume",
    "candidate task pending",
    "pending task",
    "answer the screening questionnaire",
    "screening questionnaire",
    "application will remain paused",
    "needs attention",
    "required questionnaire",
    "candidate portal task",
    "pending task is listed",
    "finish the screening questions",
    "will remain on hold",
    "documents needed",
    "requested document",
    "more information needed",
    "missing a required item",
    "requested form",
    "submit the questionnaire",
    "open task",
    "keep the process active",
    "keep the application active",
]

ACTION_MEDIUM = [
    "next steps in the process", "complete your profile",
    "verify your identity", "provide additional",
    "we need some information", "submit your",
    "need one more step", "need this before",
    "before the recruiting team can proceed",
    "before your", "can move forward",
]

UNRELATED_PHRASES = [
    "unsubscribe from all", "unsubscribe here", "unsubscribe",
    "newsletter", "marketing preferences",
    "password reset", "verify your email address",
    "confirm your email address", "account has been created",
    "subscription", "promotional", "weekly digest",
    "job alert", "job recommendations", "similar jobs",
    "jobs that match", "new jobs for you", "new jobs matching",
    "manage alerts", "manage your alerts",
    "job fair", "career fair",
    "career coach", "career insights", "career development",
    "resume tips", "resume builder", "interview preparation guide",
    "salary survey", "salary trends", "job market report",
    "networking event", "webinar",
    "virtual event", "registration is open",
    "attending the event", "does not create or change a job application",
    "referral program", "referral bonus", "refer a friend",
    "profile views", "profile was viewed",
    "update your preferences", "update your profile",
    "industry trends", "recommendations for you",
    "based on your recent activity", "based on your preferences",
    "one click", "apply with one click",
    "your dashboard", "application tracker",
    "maintenance notice", "register now", "rsvp",
    "browse open positions", "browse jobs",
    "talent community", "talent network",
    "download the report", "read the full article",
    "get started now", "try it now", "book now",
    "log in to your dashboard",
]

IN_PROCESS_PHRASES = [
    "received your application", "thank you for applying",
    "application was sent", "application has been received",
    "we have received", "successfully submitted",
    "successfully applied", "under review",
    "carefully review", "reviewing your application",
    "we will review", "being reviewed",
    "thanks for applying",
    "review your background", "review your qualifications",
    "will be in touch", "we'll reach out",
    "consider your application",
    "profile is being reviewed",
    "your materials",
    "has your materials",
    "recruiting system",
    "reviewing profiles",
    "automated confirmation",
    "possible fit",
    "someone will contact you about next steps",
    "thank you for considering us",
]


def label_email(subject: str, body: str) -> Tuple[str, float]:
    """
    Assign a category label using a scoring system.
    Each category accumulates signal strength from phrase matches;
    the strongest signal wins. This handles multi-signal emails
    (e.g., "thank you for applying... please complete the assessment")
    better than simple if/else chains.
    """
    subj = subject.lower() if isinstance(subject, str) else ""
    text = body.lower() if isinstance(body, str) else ""
    combined = subj + " " + text

    scores: Dict[str, float] = {
        "acceptance": 0.0, "rejection": 0.0, "interview": 0.0,
        "action_required": 0.0, "in_process": 0.0, "unrelated": 0.0,
    }

    # ── Rejection ──
    scores["rejection"] += _count_matches(combined, REJECTION_STRONG) * 5.0
    scores["rejection"] += _count_matches(combined, REJECTION_MEDIUM) * 2.0
    if _has_any(subj, ["status update", "application update", "follow-up"]):
        if scores["rejection"] > 0:
            scores["rejection"] += 2.0

    # ── Acceptance ──
    scores["acceptance"] += _count_matches(combined, ACCEPTANCE_PHRASES) * 5.0
    if _has_any(subj, ["offer", "offer package", "formal offer"]):
        if scores["acceptance"] > 0:
            scores["acceptance"] += 3.0

    # ── Interview ──
    scores["interview"] += _count_matches(combined, INTERVIEW_STRONG) * 4.0
    scores["interview"] += _count_matches(combined, INTERVIEW_MEDIUM) * 2.0
    if _has_any(subj, ["interview", "phone screen", "screening"]):
        scores["interview"] += 3.0

    # ── Action required ──
    scores["action_required"] += _count_matches(combined, ACTION_STRONG) * 4.0
    scores["action_required"] += _count_matches(combined, ACTION_MEDIUM) * 1.5
    if _has_any(subj, ["action required", "assessment", "complete",
                        "next steps", "action needed"]):
        scores["action_required"] += 3.0

    # ── Unrelated ──
    scores["unrelated"] += _count_matches(combined, UNRELATED_PHRASES) * 3.0
    if _has_any(subj, ["job alert", "job digest", "jobs matching",
                        "newsletter", "password reset", "verify your email",
                        "similar jobs", "weekly digest", "career insights",
                        "job market", "resume tips", "referral",
                        "job fair", "networking event", "profile views",
                        "new feature", "maintenance notice",
                        "salary survey", "industry trends"]):
        scores["unrelated"] += 3.0
    if not _has_any(combined, ["application", "applied", "position", "role",
                                "candidate", "resume", "job"]):
        scores["unrelated"] += 2.0

    # ── In-process (lower weight — this is the default) ──
    scores["in_process"] += _count_matches(combined, IN_PROCESS_PHRASES) * 1.5
    if _has_any(subj, ["profile is being reviewed", "application confirmation", "we have your materials"]):
        scores["in_process"] += 3.0

    # ── Determine winner ──
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score == 0:
        return ("in_process", 0.30)

    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]

    if margin >= 5.0:
        confidence = 0.95
    elif margin >= 3.0:
        confidence = 0.85
    elif margin >= 1.5:
        confidence = 0.70
    else:
        confidence = 0.55

    return (best_label, confidence)
