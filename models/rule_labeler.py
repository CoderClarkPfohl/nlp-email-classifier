#CS 7347 Natural Language Processing


#***Rule-based Email Labeler***

from typing import Tuple, Dict


#helper function
def _count_matches(text: str, phrases: list) -> int:
    """Count how many phrases appear in the text."""
    return sum(1 for p in phrases if p in text)

#helper function
def _has_any(text: str, phrases: list) -> bool:
    return any(p in text for p in phrases)


#reject strong dict
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
    "pursue other candidates", "move on with other applicants",
]

#reject med dict
REJECTION_MEDIUM = [
    "unfortunately", "we will not", "cannot offer",
    "no longer being considered", "not selected for",
    "other candidates whom we feel", "we regret",
    "we're unable to", "not the right fit",
]

#accptance
ACCEPTANCE_PHRASES = [
    "pleased to offer", "congratulations on your offer",
    "we are excited to extend", "offer of employment",
    "welcome aboard", "offer letter", "accepted your offer",
    "start date is", "your start date", "onboarding",
    "compensation package", "pleased to extend",
    "thrilled to welcome you", "welcome to the team",
]

#interview strong dict
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
]

#interview med dict
INTERVIEW_MEDIUM = [
    "next round", "we'd like to learn more about you",
    "meet with our team", "hiring manager would like",
    "move you to the next step", "advance to the next stage",
    "progressed to the next", "selected for the initial",
    "shortlisted", "we were impressed",
    "like to move forward with your candidacy",
]

#action strong dict
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
]

#action med dict
ACTION_MEDIUM = [
    "next steps in the process", "complete your profile",
    "verify your identity", "provide additional",
    "we need some information", "submit your",
]

#unrel dict
UNRELATED_PHRASES = [
    "unsubscribe from all", "newsletter", "marketing preferences",
    "password reset", "verify your email address",
    "confirm your email address", "account has been created",
    "subscription", "promotional", "weekly digest",
    "job alert", "job recommendations", "similar jobs",
    "jobs that match", "new jobs for you",
]

#in-process dict
IN_PROCESS_PHRASES = [
    "received your application", "thank you for applying",
    "application was sent", "application has been received",
    "we have received", "successfully submitted",
    "successfully applied", "under review",
    "carefully review", "reviewing your application",
    "we will review", "being reviewed",
    "thank you for your interest", "thanks for applying",
    "review your background", "review your qualifications",
    "will be in touch", "we'll reach out",
    "consider your application",
]


def label_email(subject: str, body: str) -> Tuple[str, float]:

    #category based on scores
    subj = subject.lower() if isinstance(subject, str) else ""
    text = body.lower() if isinstance(body, str) else ""
    combined = subj + " " + text

    scores: Dict[str, float] = {
        "acceptance": 0.0, "rejection": 0.0, "interview": 0.0,
        "action_required": 0.0, "in_process": 0.0, "unrelated": 0.0,
    }

    #rejection
    scores["rejection"] += _count_matches(combined, REJECTION_STRONG) * 5.0
    scores["rejection"] += _count_matches(combined, REJECTION_MEDIUM) * 2.0
    if _has_any(subj, ["status update", "application update", "follow-up"]):
        if scores["rejection"] > 0:
            scores["rejection"] += 2.0

    #acceptance
    scores["acceptance"] += _count_matches(combined, ACCEPTANCE_PHRASES) * 5.0

    #interview
    scores["interview"] += _count_matches(combined, INTERVIEW_STRONG) * 4.0
    scores["interview"] += _count_matches(combined, INTERVIEW_MEDIUM) * 2.0
    if _has_any(subj, ["interview", "phone screen", "screening"]):
        scores["interview"] += 3.0

    #action required
    scores["action_required"] += _count_matches(combined, ACTION_STRONG) * 4.0
    scores["action_required"] += _count_matches(combined, ACTION_MEDIUM) * 1.5
    if _has_any(subj, ["action required", "assessment", "complete",
                        "next steps", "action needed"]):
        scores["action_required"] += 3.0

    #
    scores["unrelated"] += _count_matches(combined, UNRELATED_PHRASES) * 3.0
    if not _has_any(combined, ["application", "applied", "position", "role",
                                "candidate", "resume", "job"]):
        scores["unrelated"] += 2.0

        #in-process
        scores["in_process"] += _count_matches(combined, IN_PROCESS_PHRASES) * 1.5

    #final decision based on highest score and margin
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
