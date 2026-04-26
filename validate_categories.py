#!/usr/bin/env python3
"""
Category Validation Script
==========================
Generates 200 fake emails (evenly distributed: ~33 per category),
runs them through two classifiers and compares:

  1. Rule labeler  — keyword/phrase scoring heuristic
  2. TF-IDF Ensemble (80/20 split) — SVM + LR + NB soft-voting,
     trained on 80% of the 200 emails and tested on the remaining 20%

Usage:
    python validate_categories.py
"""

import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.sparse import hstack as sp_hstack, csr_matrix

from models.rule_labeler import label_email
from models.svm_classifier import (
    build_tfidf, build_ensemble, oversample_minority
)
from models.ngram_lm import CategoryLanguageModels
from models.feature_engineering import (
    compute_keyword_features,
    compute_category_centroids, compute_cosine_features,
)
from utils.preprocessing import clean_email_body, tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random.seed(99)

# ── 200 hand-crafted emails: ~33-34 per category ──

def make_emails():
    """Return list of dicts with keys: subject, body, expected_label."""
    emails = []

    # ── ACCEPTANCE (34 emails) ──
    acceptance_emails = [
        ("Congratulations! Offer for Data Analyst", "Dear Alex, We are pleased to offer you the position of Data Analyst. Your start date is March 1, 2025. Welcome aboard!"),
        ("Offer Letter - Software Engineer at TechCo", "Dear Alex, Congratulations! We are excited to extend you an offer to join TechCo as a Software Engineer. Your compensation package includes a base salary of $120,000."),
        ("Welcome to Acme Corp!", "Dear Alex, We are thrilled to welcome you to the Acme Corp family! Your offer for the Product Manager position has been accepted. Your onboarding begins next Monday."),
        ("Your Job Offer from DataFlow", "Hi Alex, I'm delighted to inform you that you have been selected for the ML Engineer role at DataFlow. The formal offer letter is attached."),
        ("Great news from CloudBase!", "Dear Alex, I am pleased to offer you the position of DevOps Engineer at CloudBase. After meeting with our interview panel, everyone was impressed."),
        ("Job Offer: Analytics Lead", "Alex, Congratulations on your offer! We are pleased to extend an offer for the Analytics Lead position. Annual base: $135,000."),
        ("Congratulations, Alex!", "Dear Alex, It gives me great pleasure to inform you that you have been selected as our new Senior Data Scientist. Welcome aboard!"),
        ("Your Offer from FinServ", "Hi Alex, Fantastic news - we'd love to have you join us at FinServ! I'm pleased to extend a formal offer for the Quantitative Analyst position."),
        ("Offer of Employment - Backend Engineer", "Dear Alex, On behalf of NetScale, we are pleased to extend an offer of employment for the position of Backend Engineer. Start date is April 15."),
        ("Welcome to the Team!", "Dear Alex, We are excited to welcome you aboard as our new Frontend Developer. Your onboarding schedule has been prepared. Congratulations on your new role!"),
        ("Your Offer Package", "Dear Alex, Congratulations! Attached is your formal offer letter for the Site Reliability Engineer role. We are thrilled to have you join the team."),
        ("Offer Confirmation", "Hi Alex, We are pleased to confirm your offer of employment as a Cloud Architect. Your start date is set for May 1, 2025."),
        ("Congratulations on Your New Role", "Dear Alex, Welcome to the team! We are excited to extend this offer for the Data Engineer position. Please review the compensation package attached."),
        ("Job Offer - UX Designer", "Dear Alex, We are pleased to offer you the UX Designer position at DesignHub. Your salary will be $110,000 per year. Welcome aboard!"),
        ("Welcome Aboard - Product Analyst", "Hi Alex, Congratulations! We are pleased to extend you a formal offer to join our team as a Product Analyst. Start date: June 2, 2025."),
        ("Offer Extended: Research Scientist", "Dear Alex, I am pleased to extend an offer for the Research Scientist position. We believe you will be an excellent addition to our team."),
        ("Your New Position at InnoTech", "Dear Alex, Congratulations on being selected for the Technical Program Manager role at InnoTech. Your offer letter with full details is attached."),
        ("Congratulations! You Got the Job", "Hi Alex, We are thrilled to welcome you as our new Marketing Analyst! Your offer includes a base salary of $95,000 and full benefits. Welcome aboard!"),
        ("Official Offer Letter", "Dear Alex, We are pleased to offer you the position of Systems Analyst. Please sign and return the attached offer letter within 5 business days."),
        ("Offer: Solutions Architect at CloudFirst", "Alex, Congratulations! We are excited to extend an offer for the Solutions Architect position at CloudFirst. Compensation: $150,000 base."),
        ("We'd Love to Have You Join Us!", "Hi Alex, I'm pleased to extend a formal offer for the Security Analyst position. Your start date is July 1. Welcome to the team!"),
        ("Congratulations from BioGen!", "Dear Alex, On behalf of BioGen, I am pleased to offer you the Biostatistician role. We were impressed by your qualifications. Your offer letter is attached."),
        ("Your Offer Has Been Approved", "Dear Alex, Great news! Your offer for the Revenue Analyst position has been approved. Start date: August 15, 2025. Welcome aboard!"),
        ("Job Offer Notification", "Hi Alex, Congratulations! We are pleased to extend you an offer of employment as a Compliance Analyst. Please review the attached details."),
        ("Welcome to MedTech Solutions!", "Dear Alex, We are excited to welcome you aboard as our new Healthcare Data Analyst. Your onboarding packet is attached. Congratulations!"),
        ("Offer of Employment", "Dear Alex, We are pleased to offer you the Sales Engineer position. Your compensation package includes base salary plus commission. Welcome to the team!"),
        ("Congrats! Engineering Manager Offer", "Alex, I'm pleased to extend a formal offer for the Engineering Manager role. We believe you'll make a significant contribution. Welcome aboard!"),
        ("Your Offer from QuantFin", "Dear Alex, Congratulations on your offer! We are pleased to extend an offer for the Investment Analyst position at QuantFin. Salary: $125,000."),
        ("Offer Accepted - Next Steps", "Dear Alex, We are thrilled that you have accepted your offer for the AI Engineer position. Your onboarding begins on September 1. Welcome to the team!"),
        ("Formal Job Offer", "Hi Alex, I am pleased to offer you the position of Technical Writer at DocuTech. Start date is October 1, 2025. We look forward to having you!"),
        ("Your Offer Letter is Ready", "Dear Alex, Congratulations! Your offer letter for the Pricing Analyst role is ready for your review. Please sign and return within one week."),
        ("Offer Extended!", "Dear Alex, We are pleased to extend an offer of employment for the Customer Success Manager position. Your start date is November 15. Welcome!"),
        ("Great News About Your Application!", "Hi Alex, I am delighted to inform you that we are extending a formal offer for the Associate Consultant role. Salary: $90,000. Congratulations!"),
        ("Congratulations on Your Offer!", "Dear Alex, We are excited to offer you the Supply Chain Analyst position. Your compensation package is attached. Welcome aboard and we can't wait to have you!"),
    ]
    for subj, body in acceptance_emails:
        emails.append({"subject": subj, "body": body, "expected_label": "acceptance"})

    # ── REJECTION (33 emails) ──
    rejection_emails = [
        ("Update on your application", "Dear Alex, After careful consideration, we regret to inform you that we have decided not to move forward with your application at this time."),
        ("Application update", "Hi Alex, We regret to inform you that we will not be moving forward with your candidacy for this position."),
        ("Your job application status", "Alex, Thank you for your interest. At this time we're sorry to let you know we're moving forward with other candidates."),
        ("Application follow-up", "Hi Alex, Unfortunately, we have decided not to proceed with your candidacy. We have decided to move ahead with other candidates."),
        ("Regarding your application", "Dear Alex, We will not be advancing your candidacy at this time. We have chosen to pursue candidates whose experience more closely matches our needs."),
        ("Thank you for applying", "Dear Alex, Unfortunately, after careful review, we are unable to offer you a position at this time."),
        ("Application Status Update", "Dear Alex, We regret to inform you that we've decided not to move forward with your application."),
        ("An update on your application", "Hi Alex, We have decided to move forward with other applicants for this particular role. We know this isn't the news you were hoping for."),
        ("Application Decision", "Alex, After careful evaluation, we've decided not to proceed with your application. This was a competitive search."),
        ("Update: Your application", "Dear Alex, Regrettably, we will not be moving forward with your candidacy. The position has been filled by another candidate."),
        ("We appreciate your interest", "Dear Alex, After a thorough review, we regret to inform you that we are not moving forward with your application for the Data Analyst role."),
        ("Application Outcome", "Hi Alex, Unfortunately we are unable to offer you the position at this time. We received many strong candidates and this was not an easy decision."),
        ("Status of Your Application", "Dear Alex, We have decided not to move forward with your candidacy. We encourage you to apply for future openings."),
        ("Important Update", "Hi Alex, We're sorry to let you know that we will not be advancing your candidacy for the Software Engineer position."),
        ("Your Application to DataCorp", "Dear Alex, After reviewing all candidates, we regret to inform you that we have decided to pursue other applicants for this role."),
        ("Application Review Complete", "Alex, Unfortunately, we will not be proceeding with your application. We wish you the best in your career search."),
        ("Re: Your Application", "Dear Alex, We regret to inform you that we have decided not to move forward. The selection process was highly competitive."),
        ("Regarding the Analyst Position", "Hi Alex, After careful consideration, we've decided not to proceed with your candidacy. Other candidates were a better match for this specific role."),
        ("Application Notification", "Dear Alex, We regret to inform you that your application has not been selected to move forward in our hiring process."),
        ("Thank You for Your Interest", "Hi Alex, Unfortunately, we are not able to offer you a position at this time. We have decided to move ahead with other candidates."),
        ("Final Decision on Your Application", "Dear Alex, After careful deliberation, we regret that we will not be moving forward with your candidacy. We received many qualified applicants."),
        ("Application Closure", "Alex, We appreciate your interest but unfortunately we have decided not to proceed with your application for the Product Manager role."),
        ("Position Filled", "Dear Alex, We regret to inform you that the position has been filled. We will keep your resume on file for future consideration."),
        ("Regarding Your Candidacy", "Hi Alex, Unfortunately, we have decided not to move forward with your application. The decision was not easy given the strength of your background."),
        ("Update from Hiring Team", "Dear Alex, We've completed our review and regret to inform you that we will not be advancing your candidacy at this time."),
        ("Application Result", "Alex, After a thorough evaluation, we regret to inform you that we have decided not to proceed with your application."),
        ("Re: Data Scientist Application", "Dear Alex, Unfortunately, we are unable to offer you the Data Scientist position. We had many highly qualified candidates apply."),
        ("Hiring Decision", "Hi Alex, We regret to inform you that we have decided not to move forward with your candidacy. We wish you success in your job search."),
        ("Application Update - Engineer Role", "Dear Alex, After careful review, we've decided not to proceed with your application for the Engineer role. We encourage you to apply again in the future."),
        ("Thank You", "Dear Alex, We regret to inform you that we will not be moving forward with your application at this time. Thank you for your interest in our company."),
        ("Regarding Your Recent Application", "Hi Alex, Unfortunately, after careful consideration, we have decided not to proceed with your candidacy for this position."),
        ("Application Follow-Up", "Dear Alex, We appreciate your interest. Unfortunately, we have decided to move forward with other candidates for this role."),
        ("Your Application Status", "Alex, We regret to inform you that we are not moving forward with your application. The position was highly competitive."),
    ]
    for subj, body in rejection_emails:
        emails.append({"subject": subj, "body": body, "expected_label": "rejection"})

    # ── INTERVIEW (33 emails) ──
    interview_emails = [
        ("Interview Invitation: Data Analyst", "Dear Alex, We'd like to invite you for an interview for the Data Analyst position. Please reply with your availability."),
        ("Schedule your interview", "Hi Alex, Great news! We'd like to move forward with an interview. Please use the scheduling link to book your phone screen."),
        ("Next steps - Software Engineer", "Dear Alex, We'd like to schedule an initial phone screen with you. Are you available on Monday at 2pm?"),
        ("Interview Scheduled", "Hi Alex, Your interview for the Product Manager position has been scheduled. Date: March 15, 2025. Time: 10:00 AM EST."),
        ("You've been selected for an interview", "Dear Alex, We are pleased to inform you that you have been shortlisted. We would like to invite you to participate in a virtual panel interview."),
        ("Interview Request", "Hello Alex, We've reviewed your qualifications and would like to set up a call to discuss the opportunity further. Would you be available for a brief introductory call?"),
        ("Moving forward with your application", "Hi Alex, We'd like to advance your candidacy. The next step is a technical interview. Could you share your availability?"),
        ("Interview details", "Dear Alex, We are pleased to invite you to an on-site interview for the Data Scientist position. Date: April 1, 2025."),
        ("Screening call", "Hi Alex, Your profile caught our attention and we'd love to learn more. I'd like to schedule a quick screening call."),
        ("You've been selected! Next steps", "Dear Alex, A recruiter will be reaching out to you within the next 2-3 business days to schedule an initial video interview."),
        ("Phone Screen Invitation", "Hi Alex, We would like to schedule a phone screen to discuss your background and the Analyst role. Are you available this week?"),
        ("Video Interview - ML Engineer", "Dear Alex, We'd like to invite you for a video interview for the ML Engineer position. The interview will be approximately 45 minutes."),
        ("Technical Interview Invitation", "Hi Alex, We're impressed with your background and would like to invite you for a technical interview. Please select a time slot."),
        ("Interview with Hiring Manager", "Dear Alex, We'd like to schedule an interview with our hiring manager for the DevOps Engineer role. Please reply with your availability."),
        ("Panel Interview - Senior Analyst", "Hi Alex, You've been selected to participate in a panel interview with members of our analytics team. Please use the calendar link below."),
        ("Behavioral Interview Invitation", "Dear Alex, As the next step, we'd like to invite you for a behavioral interview. The session will last approximately 60 minutes."),
        ("Let's Schedule a Call", "Hi Alex, We'd like to speak with you about the Frontend Developer role. Can we schedule a 30-minute introductory call this week?"),
        ("Interview Confirmation", "Dear Alex, This email confirms your interview for the Backend Engineer position. Date: May 10. Time: 1:00 PM. Format: Video call."),
        ("Next Round - Data Engineer", "Hi Alex, Congratulations! You've been selected to move to the next round of interviews for the Data Engineer position."),
        ("Recruiter Call", "Dear Alex, I'd like to discuss your application for the Product Analyst role. Would you be available for a recruiter call this week?"),
        ("On-Site Interview Invitation", "Hi Alex, We're pleased to invite you to an on-site interview at our headquarters. Your interview day will include meeting with the team."),
        ("Interview Slot Available", "Dear Alex, We have an interview slot available for you for the Cloud Engineer position. Please book a time using the link below."),
        ("We'd Like to Meet You", "Hi Alex, After reviewing your application, we'd like to meet with you to discuss the Solutions Architect role. Can we schedule a call?"),
        ("Final Round Interview", "Dear Alex, Congratulations! You've advanced to the final round of interviews for the Engineering Manager position. Please select a date."),
        ("Interview Invitation - Analyst", "Hi Alex, We would like to invite you for an interview for the Business Analyst role. The interview will cover both technical and behavioral questions."),
        ("Schedule Your Phone Screen", "Dear Alex, We'd like to schedule a phone screen for the Research Scientist role. Please reply with times that work for you."),
        ("Virtual Interview Setup", "Hi Alex, We're setting up a virtual interview for the UX Designer position. Please confirm your availability for next week."),
        ("Meet the Team", "Dear Alex, We'd like to invite you to meet with our team for the Security Analyst role. This will be a 45-minute video call interview."),
        ("Interview Opportunity", "Hi Alex, We were impressed by your application and would like to invite you for an interview for the Technical Writer position."),
        ("Screening Interview", "Dear Alex, We'd like to schedule a screening call to learn more about your experience. Are you available for a 20-minute call this week?"),
        ("Interview Next Steps", "Hi Alex, As the next step in our process, we'd like to schedule an interview with you. Please use the booking link to select a convenient time."),
        ("We Want to Talk!", "Dear Alex, Your application for the AI Engineer role caught our attention. We'd like to invite you for an interview at your earliest convenience."),
        ("Book Your Interview", "Hi Alex, Please book your interview for the Strategy Analyst position using the scheduling link below. We look forward to speaking with you!"),
    ]
    for subj, body in interview_emails:
        emails.append({"subject": subj, "body": body, "expected_label": "interview"})

    # ── ACTION REQUIRED (33 emails) ──
    action_emails = [
        ("Action Required: Complete your assessment", "Dear Alex, Please complete the online assessment for the Data Analyst position within 5 business days. The assessment takes approximately 60 minutes."),
        ("Complete your application - Assessment Required", "Hi Alex, To continue in the hiring process, please complete the following online assessment on HackerRank. Time limit: 90 minutes."),
        ("Next Steps: Complete the coding challenge", "Dear Alex, Please complete the coding challenge via CoderPad. Both must be completed by March 30, 2025."),
        ("Assessment invitation", "Hi Alex, As part of the application process, we invite you to take our online assessment. Please complete it by April 5."),
        ("Action Needed: Complete assessment", "Hello Alex, Before we can proceed, we need you to complete a technical assessment. Estimated time: 60 minutes."),
        ("Complete Assessment - Application", "Dear Alex, You have been invited to complete an assessment. Assessment Type: Situational Judgment Test. Duration: 30 minutes."),
        ("HireVue Interview Required", "Hi Alex, Please complete a HireVue video interview as part of our selection process. You can complete this at any time before April 15."),
        ("Coding Challenge: Software Engineer", "Alex, Please complete the following coding challenge. Platform: CodeSignal. Duration: 70 minutes. Deadline: April 20."),
        ("Please complete your application", "Dear Alex, Your application is incomplete. To be considered, please complete the online application form and submit your portfolio."),
        ("Background Check Required", "Hi Alex, As part of the hiring process, we need you to complete a background check authorization. Please click the link below."),
        ("Action Required: Technical Assessment", "Dear Alex, Please complete the technical assessment for the ML Engineer role. The assessment will be available for 72 hours."),
        ("Complete Your Skills Test", "Hi Alex, We'd like you to complete a skills test for the Analyst position. Click the link below to begin. Time limit: 45 minutes."),
        ("Assessment Deadline Approaching", "Dear Alex, This is a reminder to complete your online assessment by March 25. Please log in to the assessment portal."),
        ("Pre-Employment Screening Required", "Hi Alex, Please complete the pre-employment screening as the next step in your application process."),
        ("Complete the Following Steps", "Dear Alex, To remain in consideration for the Product Manager role, please complete the following steps: upload your transcript and complete the questionnaire."),
        ("Online Test Invitation", "Hi Alex, You have been invited to take an online test for the Data Scientist position. Please complete it within 5 days."),
        ("Action Required: Submit Documents", "Dear Alex, Please submit the required documents for your application: resume, cover letter, and references."),
        ("Assessment Link Ready", "Hi Alex, Your assessment link for the DevOps Engineer position is ready. Please complete the assessment within 48 hours."),
        ("Complete Your Application", "Dear Alex, Please finish your application by completing the additional questionnaire and uploading work samples."),
        ("Coding Assessment - Next Step", "Hi Alex, As the next step, please complete the coding assessment. Platform: LeetCode. Duration: 90 minutes."),
        ("Action Required: Verify Identity", "Dear Alex, Please verify your identity by uploading a government-issued ID. This is required to proceed with your application."),
        ("Take Our Assessment", "Hi Alex, We are excited about your application. Please take our assessment to move forward. The test covers analytical skills."),
        ("Complete Required Training Module", "Dear Alex, Before your start date, please complete the required compliance training module. Deadline: April 30."),
        ("Assessment Notification", "Hi Alex, You have been selected to complete an assessment for the Engineer role. Please access it through the portal link below."),
        ("Next Steps: Complete Background Check", "Dear Alex, Please complete the background check process by providing the required information through the secure portal."),
        ("Action Needed: Application Form", "Hi Alex, We need additional information to process your application. Please complete the supplementary form by clicking the link."),
        ("Technical Test Required", "Dear Alex, As part of our evaluation, please complete the technical test. Duration: 60 minutes. Deadline: May 5, 2025."),
        ("Please Complete Your Profile", "Hi Alex, To proceed with your application for the Analyst role, please complete your candidate profile with additional details."),
        ("Action Required: Video Introduction", "Dear Alex, Please record a 2-3 minute video introduction as part of your application. Upload link is provided below."),
        ("Complete Aptitude Test", "Hi Alex, We'd like you to complete an aptitude test for the Finance Analyst role. The test takes approximately 40 minutes."),
        ("Assessment Reminder", "Dear Alex, This is a reminder to complete your online assessment for the Software Engineer position. Deadline: May 10."),
        ("Action Required: Reference Check", "Hi Alex, Please provide contact information for three professional references so we can proceed with the reference check."),
        ("Complete Your Evaluation", "Dear Alex, Please complete the online evaluation for the Project Manager role. The evaluation includes both technical and situational questions."),
    ]
    for subj, body in action_emails:
        emails.append({"subject": subj, "body": body, "expected_label": "action_required"})

    # ── IN PROCESS (33 emails) ──
    in_process_emails = [
        ("Thank you for applying", "Dear Alex, Thank you for applying to the Data Analyst position. We have received your application and our recruiting team will review your qualifications."),
        ("Application received", "Hi Alex, We've received your application for the Software Engineer position. Our hiring team is currently reviewing all applications."),
        ("Your application has been submitted", "Hello Alex, This is to confirm that your application for the Product Manager role has been successfully submitted."),
        ("Application was sent", "Alex, your application was sent to TechCo. Good luck!"),
        ("We received your application", "Dear Alex, Thank you for your interest in the Data Scientist position. We want to let you know that we have received your application materials."),
        ("Confirmation of application received", "Dear Alex, Thank you for applying to the ML Engineer role. Your application is now under review."),
        ("Thank you for your interest", "Hi Alex, Thank you for submitting your application for the Analyst role. We appreciate you taking the time to apply."),
        ("We've got your application", "Hello Alex, We've received your application for the DevOps Engineer position. Our recruiters will take a close look at your resume."),
        ("Application Acknowledged", "Dear Alex, We are writing to confirm receipt of your application for the Backend Engineer position."),
        ("Thank you for applying", "Alex, thank you for your interest in our Frontend Developer role! We have received your application and will review your background."),
        ("Your Application is Under Review", "Dear Alex, Your application for the Cloud Architect position is currently being reviewed by our hiring managers."),
        ("Application Confirmation", "Hi Alex, We confirm that your application for the Site Reliability Engineer role has been received. We will be in touch."),
        ("We're Reviewing Your Application", "Dear Alex, Thank you for applying. We are carefully reviewing your qualifications for the Data Engineer position."),
        ("Application Submitted Successfully", "Hello Alex, Your application for the UX Designer position has been successfully submitted. We'll reach out if we'd like to move forward."),
        ("Thank You for Your Application", "Dear Alex, We appreciate your interest in the Security Analyst role. Your application has been received and is under review."),
        ("Your Application Status", "Hi Alex, We wanted to let you know that your application for the Technical Writer position is being reviewed."),
        ("Application Under Consideration", "Dear Alex, Thank you for applying to the Research Scientist position. Your application is under consideration by our team."),
        ("We Received Your Resume", "Hi Alex, We have received your resume for the Business Analyst position. We will review your qualifications carefully."),
        ("Confirmation: Application Received", "Dear Alex, This email confirms that we have received your application for the Solutions Architect role. Thank you for your interest."),
        ("Application in Review", "Hi Alex, Your application for the Product Analyst position is currently in review. We will contact you if your experience matches our requirements."),
        ("Thank You!", "Dear Alex, Thank you for applying for the Engineering Manager role. We've received your application and will be reviewing it shortly."),
        ("Application Receipt Confirmation", "Hello Alex, We have received your application for the AI Engineer position. Our team will review it and get back to you."),
        ("Your Application to DataCo", "Hi Alex, Thank you for submitting your application to DataCo for the Quantitative Analyst role. We'll be in touch."),
        ("We're Processing Your Application", "Dear Alex, Your application for the Marketing Analyst position has been received and is being processed by our recruitment team."),
        ("Application Acknowledgment", "Hi Alex, We acknowledge receipt of your application for the Operations Analyst position. We will review your qualifications."),
        ("Thank You for Applying", "Dear Alex, We appreciate you taking the time to apply for the Risk Analyst position. Your application is currently under review."),
        ("Your Application Has Been Received", "Hi Alex, We have received your application for the Systems Analyst role. Our hiring team will review your materials."),
        ("Application Submitted", "Dear Alex, Thank you for your application to the Associate Consultant position. We will consider your application carefully."),
        ("We've Got Your Application!", "Hello Alex, Thanks for applying to the Strategy Analyst role! We'll review your background and experience and get back to you."),
        ("Application Received - Next Steps", "Hi Alex, We've received your application for the Investment Analyst position. Our team is reviewing all applications and will be in touch."),
        ("Thank You for Your Interest", "Dear Alex, Thank you for your interest in the Actuarial Analyst position. We have received your application and will review it thoroughly."),
        ("Confirmation of Submission", "Hi Alex, Your application for the Healthcare Data Analyst role has been submitted successfully. We will review your qualifications."),
        ("Your Application to BioTech", "Dear Alex, Thank you for applying to BioTech for the Clinical Data Analyst position. We have received your application and it is under review."),
    ]
    for subj, body in in_process_emails:
        emails.append({"subject": subj, "body": body, "expected_label": "in_process"})

    # ── UNRELATED (34 emails) ──
    unrelated_emails = [
        ("New jobs that match your profile", "Hi Alex, We found new jobs that match your profile: Data Analyst at Acme, Engineer at Beta. Unsubscribe from all alerts."),
        ("Welcome to Careers Portal", "Hi Alex, Welcome to the TechCo Careers Portal! Your account has been created. Browse open positions and set up job alerts."),
        ("Your weekly job digest", "Hi Alex, Here's your weekly roundup of jobs based on your preferences. See all recommendations. Unsubscribe."),
        ("Verify your email address", "Hi Alex, Please confirm your email address to complete your registration on our career portal. This link expires in 24 hours."),
        ("Complete your profile", "Hi Alex, Your profile on our talent network is only 60% complete. Add work experience and skills to stand out."),
        ("Job alert: Analyst positions", "Hi Alex, New jobs matching your saved search: Data Analyst at Acme (Remote). Don't miss out - apply today! Manage alerts | Unsubscribe."),
        ("Password reset request", "Hi Alex, We received a request to reset your password for your careers account. Click here to reset."),
        ("Tips for your job search", "Hi Alex, Looking for your next opportunity? Here are this week's tips: Tailor your resume, follow up within a week."),
        ("Talent Community Newsletter", "Hello Alex, Welcome to the monthly newsletter! This month: Inside look at company life, employee spotlight, upcoming events."),
        ("Similar jobs you might like", "Alex, based on your recent activity, we think you'll be interested in these roles. Apply with one click. Unsubscribe here."),
        ("New Job Recommendations", "Hi Alex, Here are some new job recommendations for you. Data Analyst at CloudCo, Engineer at DataFlow. View all jobs."),
        ("Account Created Successfully", "Hi Alex, Your account on our job board has been created successfully. You can now browse and apply for positions."),
        ("Weekly Career Insights", "Hello Alex, This week's career insights: top skills in demand, salary trends, and interview tips. Read more on our blog."),
        ("Job Fair Invitation", "Hi Alex, You're invited to our virtual job fair on April 20. Meet recruiters from top companies. Register now!"),
        ("Profile Views Update", "Hi Alex, Your profile was viewed 15 times this week. Employers are looking at your profile. Update it to increase visibility."),
        ("Subscription Confirmation", "Hi Alex, Your subscription to daily job alerts has been confirmed. You will receive job recommendations based on your preferences."),
        ("Referral Program", "Hi Alex, Earn $500 for each successful referral! Share job openings with your network. Terms and conditions apply."),
        ("New Companies Hiring", "Hello Alex, These companies are hiring now: TechCo, DataFlow, CloudBase. View open positions. Unsubscribe from this newsletter."),
        ("Career Development Webinar", "Hi Alex, Join our free webinar on career development strategies. Date: April 25. Register here."),
        ("Resume Tips", "Hello Alex, Is your resume up to date? Here are 5 tips to make your resume stand out. Read the full article."),
        ("Job Market Report", "Hi Alex, Our latest job market report is out. Key findings: tech hiring up 15%, remote positions growing. Download the report."),
        ("Networking Event", "Hello Alex, Join us for a networking event on May 5. Connect with industry professionals. RSVP now."),
        ("Salary Survey Results", "Hi Alex, The 2025 salary survey results are in. See how your compensation compares. View the full report."),
        ("Update Your Preferences", "Hi Alex, We noticed your job preferences haven't been updated recently. Update them now to get better recommendations."),
        ("New Feature: AI Resume Builder", "Hello Alex, Try our new AI-powered resume builder! Create a professional resume in minutes. Get started now."),
        ("Company Reviews", "Hi Alex, Read reviews from employees at top companies. Make informed decisions about your next career move."),
        ("Job Application Tracker", "Hello Alex, Keep track of all your applications in one place with our new application tracker feature. Try it now."),
        ("Interview Preparation Guide", "Hi Alex, Preparing for an interview? Check out our comprehensive interview preparation guide. Download now."),
        ("LinkedIn Jobs Update", "Hi Alex, 25 new jobs match your search criteria on LinkedIn. View and apply now. Manage your alerts."),
        ("Career Coach Available", "Hello Alex, Need career guidance? Our career coaches are available for free 30-minute sessions. Book now."),
        ("Job Board Maintenance Notice", "Hi Alex, Our job board will undergo maintenance on Saturday from 2-6 AM EST. We apologize for any inconvenience."),
        ("Your Application Dashboard", "Hello Alex, Log in to your dashboard to view the status of your applications. You have 3 active applications."),
        ("Industry Trends Report", "Hi Alex, Stay ahead of the curve with our latest industry trends report. Key areas: AI, cloud computing, data analytics."),
        ("Referral Bonus Reminder", "Hello Alex, Don't forget about our referral program! Refer a friend and earn up to $1000. Share now."),
    ]
    for subj, body in unrelated_emails:
        emails.append({"subject": subj, "body": body, "expected_label": "unrelated"})

    random.shuffle(emails)
    return emails


def run_ensemble_test(emails):
    """
    Train the TF-IDF ensemble on 80% of emails and test on 20%.
    Returns (per_cat_acc, overall_acc, test_results) where test_results
    is a list of dicts with keys: subject, expected, predicted, match.

    Uses ground-truth expected_label as training labels — this is the fair
    comparison against the rule labeler which also sees the same emails.
    """
    texts  = [clean_email_body(e["body"]) for e in emails]
    labels = [e["expected_label"]         for e in emails]

    # Stratified 80/20 split so every class is represented in both sets
    X_train_txt, X_test_txt, y_train, y_test, idx_train, idx_test = train_test_split(
        texts, labels, list(range(len(emails))),
        test_size=0.20, random_state=42, stratify=labels
    )

    tfidf = build_tfidf(use_stemming=True)
    X_train_tfidf = tfidf.fit_transform(X_train_txt)
    X_test_tfidf  = tfidf.transform(X_test_txt)

    # ── Enhanced NLP features (Topics 3, 2/4, 5) ──
    # N-gram LM perplexity (Topic 3) — trained on train split only
    tok_train = [tokenize(t) for t in X_train_txt]
    tok_test  = [tokenize(t) for t in X_test_txt]
    lm = CategoryLanguageModels(n=2, k=0.1)
    lm.fit(tok_train, y_train)
    lm_train = lm.perplexity_features(tok_train)
    lm_test  = lm.perplexity_features(tok_test)

    # Cosine similarity (Topics 2 & 4) — centroids from train only
    centroids, cats = compute_category_centroids(X_train_tfidf, y_train)
    cos_train = compute_cosine_features(X_train_tfidf, centroids, cats)
    cos_test  = compute_cosine_features(X_test_tfidf, centroids, cats)

    # Keyword features (Topic 5)
    kw_train = compute_keyword_features(X_train_txt)
    kw_test  = compute_keyword_features(X_test_txt)

    # Stack enhanced features with TF-IDF
    extra_train = np.hstack([lm_train, cos_train, kw_train])
    extra_test  = np.hstack([lm_test, cos_test, kw_test])
    X_train = sp_hstack([X_train_tfidf, csr_matrix(extra_train)], format="csr")
    X_test  = sp_hstack([X_test_tfidf, csr_matrix(extra_test)], format="csr")

    X_train_os, y_train_os = oversample_minority(X_train, y_train, min_samples=5)

    ens = build_ensemble()
    ens.fit(X_train_os, y_train_os)

    preds = ens.predict(X_test)
    overall_acc = accuracy_score(y_test, preds) * 100

    # Per-category accuracy on test split
    per_cat_correct = defaultdict(int)
    per_cat_total   = defaultdict(int)
    test_results = []
    for i, (expected, predicted) in enumerate(zip(y_test, preds)):
        match = expected == predicted
        per_cat_total[expected]   += 1
        per_cat_correct[expected] += int(match)
        test_results.append({
            "subject":   emails[idx_test[i]]["subject"],
            "expected":  expected,
            "predicted": predicted,
            "match":     match,
        })

    per_cat_acc = {
        cat: (per_cat_correct[cat] / per_cat_total[cat] * 100)
        for cat in per_cat_total
    }
    return per_cat_acc, overall_acc, test_results


def main():
    emails = make_emails()
    total = len(emails)
    per_cat = Counter(e["expected_label"] for e in emails)

    print("=" * 70)
    print("  CATEGORY VALIDATION: 200 fake emails")
    print("  Comparing rule labeler vs. TF-IDF ensemble (80/20 split)")
    print("=" * 70)
    print(f"\n  Total emails: {total}")
    print(f"  Distribution: {dict(sorted(per_cat.items()))}")

    # ── Rule labeler ──
    print("\n[1] Running rule labeler on all 200 emails...")
    rl_correct = 0
    rl_errors  = defaultdict(list)
    rl_confusion = defaultdict(lambda: defaultdict(int))
    results = []

    for e in emails:
        predicted, confidence = label_email(e["subject"], e["body"])
        expected = e["expected_label"]
        match = predicted == expected
        if match:
            rl_correct += 1
        else:
            rl_errors[expected].append({
                "subject":   e["subject"][:60],
                "predicted": predicted,
                "confidence": confidence,
            })
        rl_confusion[expected][predicted] += 1
        results.append({
            "subject":          e["subject"],
            "expected":         expected,
            "rule_predicted":   predicted,
            "rule_confidence":  confidence,
            "rule_match":       match,
        })

    rl_accuracy = rl_correct / total * 100
    rl_per_cat  = {
        cat: rl_confusion[cat][cat] / per_cat[cat] * 100
        for cat in per_cat
    }

    # ── Ensemble (80/20 split) ──
    print("[2] Running TF-IDF ensemble (80/20 stratified split)...")
    ens_per_cat, ens_accuracy, ens_test_results = run_ensemble_test(emails)

    # Merge ensemble predictions into results by subject
    ens_map = {r["subject"]: r for r in ens_test_results}
    for r in results:
        if r["subject"] in ens_map:
            r["ens_predicted"] = ens_map[r["subject"]]["predicted"]
            r["ens_match"]     = ens_map[r["subject"]]["match"]
        else:
            r["ens_predicted"] = None   # not in test split (was in train)
            r["ens_match"]     = None

    # ── Side-by-side accuracy table ──
    cats = sorted(per_cat.keys())
    print(f"\n  {'Category':<20s} {'Rule Labeler':>14s} {'Ensemble (20%)':>15s}")
    print(f"  {'-'*20} {'-'*14} {'-'*15}")
    for cat in cats:
        rl_a  = rl_per_cat.get(cat, 0)
        ens_a = ens_per_cat.get(cat, float("nan"))
        rl_flag  = " !" if rl_a  < 80 else ""
        ens_flag = " !" if (not np.isnan(ens_a) and ens_a < 80) else ""
        ens_str  = f"{ens_a:>10.1f}%{ens_flag}" if not np.isnan(ens_a) else f"{'n/a':>10s}"
        print(f"  {cat:<20s} {rl_a:>10.1f}%{rl_flag:<4s} {ens_str}")

    print(f"  {'-'*20} {'-'*14} {'-'*15}")
    print(f"  {'OVERALL':<20s} {rl_accuracy:>10.1f}%     {ens_accuracy:>10.1f}%")

    # ── Rule labeler confusion matrix ──
    print(f"\n  Rule Labeler Confusion Matrix (rows=expected, cols=predicted):")
    print(f"  {'':>20s}", end="")
    for lbl in cats:
        print(f" {lbl[:8]:>8s}", end="")
    print()
    for expected in cats:
        print(f"  {expected:>20s}", end="")
        for predicted in cats:
            val = rl_confusion[expected][predicted]
            marker = f" {val:>8d}" if val > 0 else f" {'·':>8s}"
            print(marker, end="")
        print()

    # ── Rule labeler errors ──
    if rl_errors:
        print(f"\n  RULE LABELER — MISLABELED EMAILS:")
        print(f"  {'-'*70}")
        for cat in sorted(rl_errors.keys()):
            print(f"\n  Expected: {cat} ({len(rl_errors[cat])} errors)")
            for err in rl_errors[cat]:
                print(f"    -> predicted={err['predicted']} (conf={err['confidence']:.2f}): {err['subject']}")

    # ── Ensemble errors ──
    ens_errors = [r for r in ens_test_results if not r["match"]]
    if ens_errors:
        print(f"\n  ENSEMBLE — MISLABELED EMAILS (test split only):")
        print(f"  {'-'*70}")
        ens_errs_by_cat = defaultdict(list)
        for r in ens_errors:
            ens_errs_by_cat[r["expected"]].append(r)
        for cat in sorted(ens_errs_by_cat.keys()):
            print(f"\n  Expected: {cat} ({len(ens_errs_by_cat[cat])} errors)")
            for err in ens_errs_by_cat[cat]:
                print(f"    -> predicted={err['predicted']}: {err['subject'][:60]}")

    # ── Save results ──
    df = pd.DataFrame(results)
    import os; os.makedirs("results", exist_ok=True)
    out_path = "results/validation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Full results saved to: {out_path}")

    # ── Summary ──
    print(f"\n{'='*70}")
    for label, acc in [("Rule labeler", rl_accuracy), ("Ensemble (20% test)", ens_accuracy)]:
        if acc >= 95:
            verdict = "PASS"
        elif acc >= 80:
            verdict = "WARN"
        else:
            verdict = "FAIL"
        print(f"  {verdict}: {label} {acc:.1f}%")
    print(f"{'='*70}")

    return rl_accuracy, rl_errors


if __name__ == "__main__":
    main()
