#!/usr/bin/env python3
"""
End-to-End Pipeline Test — 200 Fresh Balanced Emails
=====================================================
Generates 200 new emails (not the existing 2000-email training set),
runs them through the full NLP pipeline, then compares the pipeline's
predictions to the known ground-truth labels.

Usage:
    python e2e_test.py
    python e2e_test.py --output results/e2e_test
"""

import argparse
import csv
import os
import random
import tempfile
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# ── Import the full pipeline steps directly ──────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

from utils.preprocessing import clean_email_body, preprocess_for_model
from utils.entity_extraction import extract_entities
from utils.summarizer import summarize_email
from utils.excel_export import export_to_excel
from models.rule_labeler import label_email
from models.sentiment import compute_sentiment
from models.svm_classifier import train_and_evaluate

# Use a different seed from generate_synthetic_data.py (which uses 42)
# so these 200 emails are genuinely independent from the 2000-email training set.
random.seed(777)


# ─────────────────────────────────────────────────────────────────────────────
#  Mini email generator (independent from generate_synthetic_data.py)
# ─────────────────────────────────────────────────────────────────────────────

_COMPANIES = [
    "Stripe", "Figma", "Notion", "Retool", "Linear", "Vercel", "Railway",
    "Supabase", "PlanetScale", "Fly.io", "Modal", "Weights & Biases",
    "Hugging Face", "Cohere", "Mistral AI", "Perplexity", "Character AI",
    "Runway ML", "ElevenLabs", "Qdrant", "Weaviate", "Pinecone",
    "Milvus", "LangChain", "LlamaIndex", "Anyscale", "Together AI",
    "Fireworks AI", "Replicate", "OctoAI",
]
_ROLES = [
    "Machine Learning Engineer", "Platform Engineer", "Staff Engineer",
    "Growth Analyst", "Data Scientist", "Backend Engineer",
    "Product Designer", "DevRel Engineer", "Solutions Engineer",
    "Revenue Operations Analyst", "Security Engineer", "AI Researcher",
    "Infrastructure Engineer", "Developer Advocate",
]
_NAMES  = ["Jordan Kim", "Taylor Reyes", "Morgan Chen", "Casey Liu"]
_RECS   = ["Priya Sharma", "Luca Bianchi", "Mei Zhang", "Arjun Patel",
           "Sofia Andersson", "Omar Hassan"]

def _c(): return random.choice(_COMPANIES)
def _r(): return random.choice(_ROLES)
def _n(): return random.choice(_NAMES)
def _rec(): return random.choice(_RECS)
def _d():
    base = datetime(2025, 3, 1)
    return (base + timedelta(days=random.randint(0, 90))).strftime("%B %d, %Y")


TEMPLATES = {
    "acceptance": [
        lambda: (f"Your offer from {_c()}", f"Hi {_n()}, We are excited to extend an offer for the {_r()} role. Your start date is {_d()}. Salary: ${'%d,000' % random.randint(110, 180)}. Please sign and return the attached offer letter. Welcome aboard!\n\n{_rec()}\nPeople Team"),
        lambda: (f"Congratulations — {_c()} offer", f"Dear {_n()}, Congratulations! We are pleased to offer you the {_r()} position at {_c()}. We were impressed throughout the process. Your compensation package is attached. Start date: {_d()}. Welcome to the team!\n\n{_rec()}"),
        lambda: (f"Offer letter: {_r()}", f"{_n()}, We are thrilled to welcome you aboard as our {_r()}. Your offer letter is attached. Base salary, equity, and benefits details are enclosed. Onboarding begins {_d()}.\n\n{_rec()}, Talent"),
        lambda: (f"We'd love to have you join {_c()}", f"Hi {_n()}, Great news — we are extending a formal offer for the {_r()} role. Compensation: ${random.randint(120, 200)}k base + equity. Start date: {_d()}. Please accept by replying to this email.\n\n{_rec()}"),
        lambda: (f"Offer extended: {_r()} at {_c()}", f"Dear {_n()}, On behalf of {_c()}, I am pleased to offer you the position of {_r()}. The attached offer letter contains all the details including your start date of {_d()} and compensation package. We look forward to having you on the team!"),
    ],
    "rejection": [
        lambda: (f"Your application to {_c()}", f"Hi {_n()}, Thank you for your interest in the {_r()} role at {_c()}. After careful consideration, we regret to inform you that we will not be moving forward with your application. We wish you the best in your search.\n\n{_rec()}"),
        lambda: (f"Update on your candidacy", f"Dear {_n()}, Unfortunately, after a thorough review, we have decided not to proceed with your candidacy for the {_r()} position. The decision was difficult given the competitive pool. Thank you for your time.\n\n{_rec()}, {_c()}"),
        lambda: (f"Application status — {_c()}", f"{_n()}, Thank you for applying to {_c()} for the {_r()} role. We regret to inform you that we are not moving forward with your application at this time. We encourage you to apply for future openings."),
        lambda: (f"Re: {_r()} application", f"Hi {_n()}, We've completed our review and are moving forward with other candidates for the {_r()} role. This was not an easy decision — we had many strong candidates. We will keep your profile for future opportunities.\n\n{_rec()}"),
        lambda: (f"Decision on your {_c()} application", f"Dear {_n()}, We regret to inform you that we have decided not to advance your candidacy for the {_r()} position. We had a highly competitive applicant pool and had to make some tough choices. Best of luck in your search."),
    ],
    "interview": [
        lambda: (f"Interview invitation: {_r()}", f"Hi {_n()}, We reviewed your application for the {_r()} role and would like to invite you for an interview. The session will be approximately 45 minutes. Please use the link to book a time: [Calendly link]\n\n{_rec()}, {_c()} Recruiting"),
        lambda: (f"Let's schedule a call — {_c()}", f"Dear {_n()}, I am reaching out to schedule an introductory call to discuss the {_r()} opportunity at {_c()}. Could you share your availability this week? This will be a 30-minute phone screen.\n\n{_rec()}"),
        lambda: (f"Technical interview — {_r()} at {_c()}", f"{_n()}, Congrats! You've been selected for a technical interview for the {_r()} position. Date: {_d()}. The interview will cover system design and coding. A calendar invite is attached.\n\n{_rec()}"),
        lambda: (f"Next step: interview at {_c()}", f"Hi {_n()}, We are excited to invite you to the next round — a video interview for the {_r()} role. The session is scheduled for {_d()}. Please confirm your attendance by replying to this email.\n\n{_rec()}"),
        lambda: (f"Schedule your interview — {_c()}", f"Dear {_n()}, Thank you for your application! We would like to move forward and invite you to a panel interview for the {_r()} position. Please select a time that works for you via the booking link below. We look forward to speaking with you!"),
    ],
    "action_required": [
        lambda: (f"Action required: complete your assessment", f"Hi {_n()}, To continue in the application process for the {_r()} role at {_c()}, please complete the online technical assessment. Platform: HackerRank. Time limit: 75 minutes. Deadline: {_d()}.\n\nGood luck!"),
        lambda: (f"Complete your coding challenge — {_c()}", f"Dear {_n()}, As the next step for the {_r()} position, please complete the coding challenge on CodeSignal. The challenge covers data structures and algorithms. You have 5 days to complete it. Link: [CodeSignal]\n\n{_rec()}"),
        lambda: (f"Assessment invitation from {_c()}", f"{_n()}, You've been selected to move forward! Please complete the following assessment for the {_r()} role: Technical Skills Test (60 minutes). Deadline: {_d()}. Note: once started, the timer cannot be paused."),
        lambda: (f"Next steps: please complete the following", f"Hi {_n()}, We are excited about your application for the {_r()} role. Before proceeding, please complete these steps: (1) Technical assessment via HireVue, (2) Upload latest work samples. Deadline: {_d()}."),
        lambda: (f"Background check required — {_c()}", f"Dear {_n()}, As part of the hiring process for the {_r()} position, we need you to complete a background check authorization. Please click the link below to provide the required information by {_d()}. Contact hr@company.com with questions."),
    ],
    "in_process": [
        lambda: (f"We received your application — {_c()}", f"Hi {_n()}, Thank you for applying to the {_r()} position at {_c()}. We have received your application and our recruiting team will review your qualifications carefully. We will be in touch if we'd like to move forward.\n\n{_c()} Talent"),
        lambda: (f"Application confirmed: {_r()}", f"Dear {_n()}, This is to confirm that your application for the {_r()} role at {_c()} has been successfully submitted. Our team is reviewing all applications and will reach out to candidates who are a strong fit.\n\n{_rec()}"),
        lambda: (f"Thanks for applying to {_c()}", f"{_n()}, Thank you for your interest in the {_r()} role! We have received your application and appreciate you taking the time to apply. If your background aligns with our requirements, someone from our team will reach out."),
        lambda: (f"Application under review", f"Hi {_n()}, We wanted to let you know that your application for the {_r()} position at {_c()} is currently under review by our hiring team. We will contact you regarding next steps. Thank you for your patience."),
        lambda: (f"Your application to {_c()} — submitted", f"Dear {_n()}, Your application for the {_r()} position has been submitted successfully. We review all applications carefully and will reach out if there is a match. Thank you for considering {_c()} as your next career opportunity."),
    ],
    "unrelated": [
        lambda: (f"New jobs matching your search", f"Hi {_n()}, Here are new jobs that match your search: {_r()} at {_c()}, {_r()} at {_c()}, {_r()} at {_c()}. Apply with one click. Manage your job alerts. Unsubscribe from all alerts."),
        lambda: (f"Your weekly job digest", f"Hello {_n()}, Here's your weekly roundup of recommended jobs. {_c()} is hiring! See all jobs on your dashboard. Update your preferences to get better recommendations. Unsubscribe here."),
        lambda: (f"Verify your email — {_c()} careers portal", f"Hi {_n()}, Please confirm your email address to complete your registration on {_c()}'s career portal. Click here to verify your email. This link expires in 24 hours."),
        lambda: (f"Password reset request", f"Hi {_n()}, We received a request to reset your password for your careers account at {_c()}. Click here to reset: [Reset Link]. If you did not request this, ignore this email. Security Team."),
        lambda: (f"Talent community newsletter — {_c()}", f"Hello {_n()}, Welcome to {_c()}'s monthly talent community newsletter. This month: company news, open roles, employee spotlights, upcoming events. Unsubscribe from this newsletter."),
    ],
}


def generate_200_emails():
    """
    Generate exactly 200 emails: 33 or 34 per category.
    Uses seed 777 — independent from the 2000-email training corpus (seed 42).
    """
    emails = []
    counts = {cat: 34 if i < 2 else 33
              for i, cat in enumerate(TEMPLATES)}  # 34+34+33+33+33+33 = 200

    for cat, n in counts.items():
        fns = TEMPLATES[cat]
        for _ in range(n):
            fn = random.choice(fns)
            subject, body = fn()
            emails.append({
                "company": random.choice(_COMPANIES),
                "subject": subject,
                "email_body": body,
                "date_only": (datetime(2025, 1, 1) + timedelta(days=random.randint(0, 120))).strftime("%Y-%m-%d"),
                "true_label": cat,
            })

    random.shuffle(emails)
    return emails


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run every stage of the NLP pipeline on the DataFrame."""
    # Stage 1: preprocess
    df["clean_body"] = df["email_body"].apply(clean_email_body)
    df["model_body"]  = df["email_body"].apply(preprocess_for_model)

    # Stage 2: rule labeler
    results = df.apply(
        lambda r: label_email(str(r.get("subject", "")), str(r.get("email_body", ""))),
        axis=1,
    )
    df["rule_label"]      = [r[0] for r in results]
    df["rule_confidence"] = [r[1] for r in results]

    # Stage 3: sentiment
    sents = df["clean_body"].apply(compute_sentiment)
    df["sentiment_label"]    = [s["label"]    for s in sents]
    df["sentiment_compound"] = [s["compound"] for s in sents]

    # Stage 4: entity extraction
    entities = df.apply(
        lambda r: extract_entities(str(r.get("email_body", "")), str(r.get("company", ""))),
        axis=1,
    )
    df["extracted_role"]  = [e["job_role"]        for e in entities]
    df["contact_person"]  = [e["contact_person"]  for e in entities]
    df["contact_email"]   = [e["contact_email"]   for e in entities]
    df["dates_mentioned"] = ["; ".join(e["dates_mentioned"]) for e in entities]

    # Stage 5: summarization
    df["summary"] = df["clean_body"].apply(lambda t: summarize_email(t, max_sentences=2))

    # Stage 6: ensemble classifier (train on rule_label, CV on same data)
    import numpy as np
    tfidf, ensemble, cv_preds, cv_probas, metrics = train_and_evaluate(
        df["clean_body"].tolist(), df["rule_label"].tolist(), n_folds=5
    )
    df["final_label"]          = cv_preds
    df["ensemble_confidence"]  = cv_probas.max(axis=1).round(4)
    sorted_p = np.sort(cv_probas, axis=1)
    df["ensemble_top2_gap"]    = (sorted_p[:, -1] - sorted_p[:, -2]).round(4)

    return df, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="200-email end-to-end pipeline test")
    parser.add_argument("--output", default="results/e2e_test",
                        help="Output directory (default: results/e2e_test)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 65)
    print("  END-TO-END PIPELINE TEST — 200 Fresh Balanced Emails")
    print("=" * 65)

    # ── Generate emails ──────────────────────────────────────────
    print("\n[1] Generating 200 fresh emails (seed=777, independent of training set)...")
    emails = generate_200_emails()
    df = pd.DataFrame(emails)

    dist = dict(sorted(Counter(df["true_label"]).items()))
    print(f"     Total: {len(df)} emails")
    print(f"     Distribution: {dist}")

    # ── Run pipeline ─────────────────────────────────────────────
    print("\n[2] Running full NLP pipeline...")
    df, metrics = run_pipeline(df)

    # ── Accuracy vs ground truth ──────────────────────────────────
    true = df["true_label"].tolist()
    cats = sorted(set(true))

    rule_acc  = accuracy_score(true, df["rule_label"])
    final_acc = accuracy_score(true, df["final_label"])

    print("\n" + "=" * 65)
    print("  ACCURACY vs GROUND TRUTH")
    print("=" * 65)
    print(f"\n  {'Stage':<35s} {'Accuracy':>10s}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'Majority baseline':.<35s} {Counter(true).most_common(1)[0][1]/len(true)*100:>9.1f}%")
    print(f"  {'Rule labeler':.<35s} {rule_acc*100:>9.1f}%")
    print(f"  {'Ensemble (5-fold CV)':.<35s} {final_acc*100:>9.1f}%")

    print(f"\n  Per-category breakdown:")
    print(f"\n  {'Category':<22s} {'Rule':>8s} {'Ensemble':>10s} {'N':>5s}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*5}")
    for cat in cats:
        mask = [t == cat for t in true]
        cat_true  = [t for t, m in zip(true, mask) if m]
        cat_rule  = [p for p, m in zip(df["rule_label"].tolist(), mask) if m]
        cat_final = [p for p, m in zip(df["final_label"].tolist(), mask) if m]
        r_acc = accuracy_score(cat_true, cat_rule)  * 100
        f_acc = accuracy_score(cat_true, cat_final) * 100
        flag  = " !" if f_acc < 70 else ""
        print(f"  {cat:<22s} {r_acc:>7.1f}% {f_acc:>9.1f}%{flag}  {len(cat_true):>4d}")

    print(f"\n  Ensemble classification report (vs true labels):")
    print(classification_report(true, df["final_label"], labels=cats, zero_division=0))

    # ── Save outputs ──────────────────────────────────────────────
    out_cols = [
        "company", "subject", "date_only", "true_label",
        "rule_label", "rule_confidence", "final_label",
        "ensemble_confidence", "sentiment_label", "sentiment_compound",
        "extracted_role", "contact_person", "summary",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    csv_path  = os.path.join(args.output, "e2e_classified.csv")
    xlsx_path = os.path.join(args.output, "e2e_classified.xlsx")

    df[out_cols].to_csv(csv_path, index=False)
    export_to_excel(df[out_cols], xlsx_path, label_column="final_label")

    print(f"\n  Outputs saved to {args.output}/")
    print(f"    {csv_path}")
    print(f"    {xlsx_path}")

    # ── Final verdict ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    if final_acc >= 0.90:
        verdict = "PASS — pipeline working correctly (≥90%)"
    elif final_acc >= 0.80:
        verdict = "WARN — acceptable but check flagged categories (≥80%)"
    else:
        verdict = "FAIL — significant classification errors (<80%)"
    print(f"  {verdict}")
    print(f"  Rule labeler: {rule_acc*100:.1f}%  |  Ensemble: {final_acc*100:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
