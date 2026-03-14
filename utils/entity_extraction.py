"""
Entity extraction from job application emails.
Detects company names, job roles, contact persons, dates, and addresses
using regex and heuristic patterns.
"""

import re
from typing import Dict, List, Optional


def extract_job_role(text: str) -> Optional[str]:
    """
    Extract the job role/title from the email body.
    Looks for common patterns like "applied for the X position" or
    "application for X".
    """
    patterns = [
        r"(?:applied|applying|application)\s+(?:for|to)\s+(?:the\s+)?(.{5,80}?)\s+(?:position|role|opening|job|opportunity)",
        r"(?:position|role)\s+(?:of|:)\s+(.{5,80?}?)(?:\s+at|\s+with|\.|,|\n)",
        r"interest in (?:the\s+)?(.{5,80}?)\s+(?:position|role|opening)",
        r"for the (.{5,80}?)\s+(?:position|role)\s+at",
        r"application for\s+(.{5,80?})(?:\s+received|\s+has been)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            role = match.group(1).strip().strip(".,;:'\"")
            # Filter out overly long or garbage matches
            if 3 < len(role) < 80:
                return role
    return None


def extract_dates(text: str) -> List[str]:
    """
    Extract date-like strings from the email text.
    Returns a list of raw date strings found.
    """
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]
    dates = []
    for pat in patterns:
        dates.extend(re.findall(pat, text, re.IGNORECASE))
    return dates


def extract_contact_person(text: str) -> Optional[str]:
    """
    Attempt to find a recruiter or contact person name.
    Looks for patterns like 'contact X', 'recruiter: X', 'reach out to X'.
    """
    patterns = [
        r"(?:recruiter|hiring manager|contact)\s*(?:is|:)\s*([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"(?:reach out to|contact)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"(?:sincerely|regards|best|cheers),?\s*\n\s*([A-Z][a-z]+\s+[A-Z][a-z]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            name = match.group(1).strip()
            if 3 < len(name) < 40:
                return name
    return None


def extract_contact_email(text: str) -> Optional[str]:
    """Extract a contact email address from the body (not the sender)."""
    # Find all emails, try to skip noreply-type addresses
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    for em in emails:
        lower = em.lower()
        if not any(skip in lower for skip in ["noreply", "no-reply", "donotreply",
                                                "mailer", "notification", "auto"]):
            return em
    return emails[0] if emails else None


def extract_entities(text: str, company: str = "") -> Dict:
    """
    Run all entity extractors on a single email body.

    Parameters
    ----------
    text : str
        Raw or lightly cleaned email body.
    company : str
        Company name if already known from the dataset.

    Returns
    -------
    dict with keys: company, job_role, contact_person, contact_email, dates
    """
    return {
        "company": company if company else None,
        "job_role": extract_job_role(text),
        "contact_person": extract_contact_person(text),
        "contact_email": extract_contact_email(text),
        "dates_mentioned": extract_dates(text),
    }
