#!/usr/bin/env python3


#Synthetic Data Generator
#Creating emails that later would be used for training.


import random
import csv
import os
from datetime import datetime, timedelta

random.seed(42)


#list of random company names
COMPANIES = [
    "Amazon", "Google", "Microsoft", "Apple", "Meta", "Netflix", "Tesla",
    "Salesforce", "Adobe", "Oracle", "Intel", "IBM", "Cisco", "VMware",
    "Workday", "Snowflake", "Databricks", "Palantir", "Stripe", "Square",
    "Affirm", "Coinbase", "Robinhood", "Spotify", "Uber", "Lyft", "Airbnb",
    "DoorDash", "Instacart", "Pinterest", "Snap", "Twitter", "LinkedIn",
    "GitHub", "Atlassian", "Slack", "Zoom", "Twilio", "Cloudflare",
    "CrowdStrike", "Okta", "MongoDB", "Elastic", "HashiCorp", "Confluent",
    "Datadog", "Splunk", "ServiceNow", "Palo Alto Networks", "Fortinet",
    "Walmart", "Target", "Costco", "Home Depot", "JPMorgan Chase",
    "Goldman Sachs", "Morgan Stanley", "Bank of America", "Wells Fargo",
    "Citi", "Capital One", "American Express", "Visa", "Mastercard",
    "PayPal", "Fidelity", "BlackRock", "Vanguard", "Charles Schwab",
    "Deloitte", "PwC", "EY", "KPMG", "McKinsey", "BCG", "Bain",
    "Accenture", "Booz Allen Hamilton", "Lockheed Martin", "Boeing",
    "Raytheon", "Northrop Grumman", "General Electric", "Honeywell",
    "3M", "Johnson & Johnson", "Pfizer", "Moderna", "AbbVie",
    "UnitedHealth Group", "Anthem", "CVS Health", "Humana",
    "Procter & Gamble", "Unilever", "Coca-Cola", "PepsiCo",
    "Nike", "Adidas", "Samsung", "Sony", "LG", "Siemens",
    "Autodesk", "Figma", "Canva", "Notion", "Airtable", "Asana",
    "Monday.com", "HubSpot", "Zendesk", "Freshworks", "Intercom",
    "Grammarly", "Duolingo", "Coursera", "edX", "Udemy",
    "SpaceX", "Blue Origin", "Rivian", "Lucid Motors", "Waymo",
    "Aurora", "Nuro", "Scale AI", "Anthropic", "OpenAI", "Cohere",
    "Hugging Face", "Stability AI", "Midjourney",
    "Medtronic", "Stryker", "Boston Scientific", "Intuitive Surgical",
    "Epic Systems", "Cerner", "Veeva Systems", "IQVIA",
    "Thermo Fisher", "Danaher", "Agilent", "Waters Corporation",
]

#list of roles
ROLES = [
    "Data Analyst", "Data Scientist", "Business Analyst",
    "Business Intelligence Analyst", "Machine Learning Engineer",
    "Software Engineer", "Senior Software Engineer", "Full Stack Developer",
    "Frontend Engineer", "Backend Engineer", "DevOps Engineer",
    "Cloud Engineer", "Site Reliability Engineer", "Data Engineer",
    "Analytics Engineer", "Product Manager", "Product Analyst",
    "Research Scientist", "Applied Scientist", "AI Engineer",
    "Quantitative Analyst", "Financial Analyst", "Operations Analyst",
    "Marketing Analyst", "Supply Chain Analyst", "Risk Analyst",
    "Compliance Analyst", "Systems Analyst", "Security Analyst",
    "Solutions Architect", "Technical Program Manager",
    "Engineering Manager", "UX Researcher", "UX Designer",
    "Technical Writer", "Sales Engineer", "Customer Success Manager",
    "Associate Consultant", "Strategy Analyst", "Investment Analyst",
    "Actuarial Analyst", "Biostatistician", "Clinical Data Analyst",
    "Healthcare Data Analyst", "Revenue Analyst", "Pricing Analyst",
]

#list of applicant names
APPLICANT_NAMES = [
    "Michael Gary Scott", "Micheal Gary Scott",  # both spellings from real data
]

#list of recruiter names
RECRUITER_NAMES = [
    "Sarah Johnson", "James Williams", "Emily Chen", "David Rodriguez",
    "Maria Garcia", "Robert Kim", "Jennifer Lee", "Thomas Brown",
    "Amanda Davis", "Christopher Martinez", "Rachel Thompson",
    "Daniel Wilson", "Michelle Anderson", "Kevin Taylor", "Laura White",
    "Brian Harris", "Stephanie Clark", "Andrew Lewis", "Nicole Robinson",
    "Matthew Walker", "Jessica Hall", "Ryan Allen", "Samantha Young",
]


def rand_name():
    return random.choice(APPLICANT_NAMES)

def rand_company():
    return random.choice(COMPANIES)

def rand_role():
    return random.choice(ROLES)

def rand_recruiter():
    return random.choice(RECRUITER_NAMES)

def rand_date():
    base = datetime(2025, 1, 1)
    offset = timedelta(days=random.randint(0, 120))
    return base + offset

def rand_job_id():
    #generating rnadom job id
    return f"{random.randint(100000, 999999)}"

def rand_sender(company):
    patterns = [
        f"{company} Careers <noreply@{company.lower().replace(' ', '')}.com>",
        f"Talent Acquisition <careers@{company.lower().replace(' ', '')}.com>",
        f"no-reply@{company.lower().replace(' ', '')}.com",
        f"{company} HR <hr@{company.lower().replace(' ', '')}.com>",
        f"recruiting@{company.lower().replace(' ', '')}.com",
    ]
    return random.choice(patterns)



#Randomly generated emails under "IN PROCESS"
def gen_in_process():
    c = rand_company()
    r = rand_role()
    n = rand_name()
    jid = rand_job_id()

    templates = [
        {
            "subject": f"Thank you for applying to {c}!",
            "body": f"Dear {n},\n\nThank you for applying to the {r} position at {c}. We have received your application and our recruiting team will review your qualifications carefully.\n\nIf your background aligns with our requirements, a member of our team will reach out to discuss next steps. Please note that due to the high volume of applications, this process may take several weeks.\n\nIn the meantime, you can track your application status through our careers portal.\n\nBest regards,\n{c} Talent Acquisition",
        },
        {
            "subject": f"Application received: {r} at {c}",
            "body": f"Hi {n},\n\nWe've received your application for the {r} position (Job ID: {jid}) at {c}. Thank you for your interest in joining our team!\n\nOur hiring team is currently reviewing all applications. We'll be in touch if we'd like to move forward.\n\nWe appreciate your patience during this process.\n\nRegards,\nThe {c} Recruiting Team",
        },
        {
            "subject": f"Your application to {c} has been submitted",
            "body": f"Hello {n},\n\nThis is to confirm that your application for the {r} role at {c} has been successfully submitted.\n\nWhat happens next:\n- Our team will carefully review your resume and qualifications\n- If your experience matches our requirements, we will contact you\n- You can check your application status at any time through our portal\n\nThank you for considering {c} as your next career step.\n\nBest,\n{c} Careers",
        },
        {
            "subject": f"{n}, your application was sent to {c}",
            "body": f"Your application was sent to {c}\n\n{n} your application was sent to {c}\n\n{c}\n{r}\n\nYour application was sent to the employer. Good luck!\n\nNote: This email confirms your application was submitted.",
        },
        {
            "subject": f"We received your application for {r}",
            "body": f"Dear {n},\n\nThank you for your interest in the {r} position at {c}. We want to let you know that we have received your application materials.\n\nOur recruiting team reviews every application thoroughly. We will be in contact with you regarding the status of your application.\n\nPlease do not reply to this automated message.\n\nSincerely,\n{c} Human Resources",
        },
        {
            "subject": f"Confirmation of application received for {r} - {jid}",
            "body": f"This notification was automatically generated.\n\nDear {n},\n\nThank you for applying to the role of {jid} - {r} at {c}. We're excited that you're considering us as part of your career journey.\n\nYour application is now under review. We strive to provide updates within 2-3 weeks.\n\nWarm regards,\nThe Talent Team at {c}",
        },
        {
            "subject": f"Thank you for your interest in {c}",
            "body": f"Hi {n},\n\nThank you for submitting your application for the {r} role at {c}. We appreciate you taking the time to apply.\n\nOur team is reviewing applications and will reach out to candidates who are a good match. We will keep your resume on file for future openings as well.\n\nThank you again,\n{rand_recruiter()}\nRecruiter, {c}",
        },
        {
            "subject": f"{r} at {c}: we've got your application",
            "body": f"Hello {n},\n\nWe've received your application for the {r} position at {c}. Thank you for taking the time to apply and for considering joining us.\n\nFirst of all, our recruiters will take a close look at your resume and cover letter. If your qualifications are a good match for the role, you'll hear from us soon.\n\nIn the meantime, feel free to explore other opportunities on our careers page.\n\nBest regards,\nThe {c} Talent Team",
        },
        {
            "subject": f"Application Acknowledged - {r}",
            "body": f"Dear {n},\n\nWe are writing to confirm receipt of your application for the {r} position at {c} (Ref: {jid}).\n\nYour application is being reviewed by our hiring managers. We will notify you of any changes to your application status.\n\nKind regards,\n{c} Recruitment",
        },
        {
            "subject": f"Thank you for applying at {c}",
            "body": f"{n}, thank you for your interest in our {r} role at {c}! We have received your application and are delighted that you would consider joining our team.\n\nWe will review your background and experience and get back to you if there is a potential fit. Due to the volume of applications we receive, only selected candidates will be contacted.\n\nBest wishes,\n{c} Hiring Team",
        },
    ]

    t = random.choice(templates)
    return c, t["subject"], t["body"], "in_process"

#Randomly generated emails under "REJECTION"
def gen_rejection():
    c = rand_company()
    r = rand_role()
    n = rand_name()

    templates = [
        {
            "subject": f"Update on your {c} application",
            "body": f"Dear {n},\n\nThank you for your interest in the {r} position at {c} and for taking the time to apply.\n\nAfter careful consideration, we regret to inform you that we have decided not to move forward with your application at this time. We received many qualified applicants and this was a difficult decision.\n\nWe encourage you to apply for future openings that match your skills and experience. We wish you the best in your career search.\n\nSincerely,\n{c} Talent Acquisition",
        },
        {
            "subject": f"{c} | Application update",
            "body": f"Hi {n},\n\nThank you for your interest in the {r} role. We regret to inform you that we will not be moving forward with your candidacy for this position.\n\nWe encourage you to visit our website at careers.{c.lower().replace(' ', '')}.com to view other opportunities that may align with your experience.\n\nBest regards,\n{c} Recruiting",
        },
        {
            "subject": f"Your job application status",
            "body": f"{c}\n\n{n},\n\nThank you for your interest in the {r} position at {c}. At this time we're sorry to let you know we're moving forward with other candidates.\n\nPlease continue to visit our careers page for future opportunities.\n\n{c} Talent Acquisition Team",
        },
        {
            "subject": f"Application follow-up: {c}",
            "body": f"Hi {n},\n\nThank you for your interest in {c}! Unfortunately, we have decided not to proceed with your candidacy for the current {r} opening. We received many qualified applicants and have decided to move ahead with other candidates whom we feel are a better match for this specific role.\n\nThis does not reflect on your qualifications, and we encourage you to apply again in the future.\n\nWishing you the best,\n{rand_recruiter()}\n{c}",
        },
        {
            "subject": f"Regarding your application to {c}",
            "body": f"Dear {n},\n\nWe appreciate your interest in joining {c} and the time you invested in applying for the {r} position.\n\nAfter a thorough review of your application, we have determined that we will not be advancing your candidacy at this time. The selection process was highly competitive, and while your background is impressive, we have chosen to pursue candidates whose experience more closely matches our current needs.\n\nWe will keep your information on file and may reach out if a suitable opportunity arises.\n\nWith best wishes,\n{c} HR Team",
        },
        {
            "subject": f"Thank you for applying to {c}",
            "body": f"Dear {n},\n\nThank you very much for your recent application to the {r} position at {c}.\n\nUnfortunately, after careful review, we are unable to offer you a position at this time. We had many strong candidates apply and it was not an easy decision.\n\nWe genuinely appreciate your interest in {c} and hope you'll consider us for future opportunities.\n\nBest regards,\n{rand_recruiter()}\nTalent Acquisition, {c}",
        },
        {
            "subject": f"Application Status Update - {r}",
            "body": f"Dear {n},\n\nThank you for applying for the {r} position at {c}. At this time, we regret to inform you that we've decided not to move forward with your application.\n\nAs a growing organization we always have new positions available and therefore we encourage you to check our careers page regularly.\n\nWe appreciate your interest and wish you success in your job search.\n\nSincerely,\nRecruitment Team\n{c}",
        },
        {
            "subject": f"An update on your application",
            "body": f"Hi {n},\n\nWe wanted to follow up regarding your application for {r} at {c}.\n\nAfter reviewing all candidates, we have decided to move forward with other applicants for this particular role. We know this isn't the news you were hoping for, and we appreciate the effort you put into your application.\n\nPlease don't hesitate to apply for other positions at {c} in the future.\n\nAll the best,\n{c} Recruiting",
        },
        {
            "subject": f"{c} - Application Decision",
            "body": f"{n},\n\nThank you for your time and interest in the {r} role at {c}. After careful evaluation, we've decided not to proceed with your application. This was a competitive search and we had to make some tough choices.\n\nYour resume will remain in our database for future consideration.\n\nRegards,\n{c} Talent Team",
        },
        {
            "subject": f"Update: Your application for {r}",
            "body": f"Dear {n},\n\nWe have completed our review of applications for the {r} position at {c}. We appreciate your interest in this opportunity.\n\nRegrettably, we will not be moving forward with your candidacy. The position has been filled by another candidate.\n\nWe wish you all the best in your career endeavors and encourage you to monitor our openings for future roles.\n\nBest,\n{rand_recruiter()}\n{c}",
        },
    ]

    t = random.choice(templates)
    return c, t["subject"], t["body"], "rejection"

#Randomly generated emails under "INTERVIEW"
def gen_interview():
    c = rand_company()
    r = rand_role()
    n = rand_name()
    rec = rand_recruiter()
    d = rand_date()
    date_str = d.strftime("%B %d, %Y")
    time_str = random.choice(["10:00 AM", "11:30 AM", "1:00 PM", "2:30 PM", "3:00 PM", "4:00 PM"])

    templates = [
        {
            "subject": f"Interview Invitation: {r} at {c}",
            "body": f"Dear {n},\n\nThank you for your interest in the {r} position at {c}. We were impressed by your background and would like to invite you for an interview.\n\nWe'd like to schedule a video interview at your earliest convenience. Please reply to this email with your availability for the coming week, or use the scheduling link below to book a time slot.\n\nThe interview will be approximately 45-60 minutes and will be conducted by {rec}, our hiring manager.\n\nWe look forward to speaking with you!\n\nBest regards,\n{rec}\n{c} Talent Acquisition",
        },
        {
            "subject": f"Schedule your interview with {c}",
            "body": f"Hi {n},\n\nGreat news! After reviewing your application for the {r} role, we'd like to move forward with an interview.\n\nPlease use the following link to schedule your interview:\n[Scheduling Link]\n\nThe phone screen will last approximately 30 minutes and will cover your background, experience, and interest in the role.\n\nIf you have any questions, feel free to reach out to me directly.\n\nBest,\n{rec}\nRecruiter, {c}",
        },
        {
            "subject": f"Next steps - {r} position at {c}",
            "body": f"Dear {n},\n\nI hope this email finds you well. I'm reaching out regarding your application for the {r} position at {c}.\n\nI'm pleased to let you know that our team has reviewed your application and we'd like to schedule an initial phone screen with you. This will be a 30-minute conversation to learn more about your experience and share details about the role.\n\nAre you available on {date_str} at {time_str} EST? If not, please suggest some alternative times.\n\nLooking forward to connecting!\n\n{rec}\nTalent Acquisition Partner\n{c}",
        },
        {
            "subject": f"Interview Scheduled - {r} at {c}",
            "body": f"Hi {n},\n\nYour interview for the {r} position at {c} has been scheduled.\n\nDate: {date_str}\nTime: {time_str} EST\nFormat: Video call (link will be sent separately)\nInterviewer: {rec}\nDuration: 60 minutes\n\nPlease confirm your attendance by replying to this email. If you need to reschedule, let us know at least 24 hours in advance.\n\nGood luck!\n\n{c} Recruiting Team",
        },
        {
            "subject": f"Congratulations! You've been selected for an interview at {c}",
            "body": f"Dear {n},\n\nWe are pleased to inform you that you have been shortlisted for the {r} position at {c}.\n\nWe would like to invite you to participate in a virtual panel interview with members of our {random.choice(['engineering', 'analytics', 'data science', 'product'])} team. The interview will consist of both technical and behavioral questions.\n\nPlease select a time slot that works for you using the calendar link below.\n\nWe look forward to meeting you!\n\nWarm regards,\n{rec}\n{c} People Operations",
        },
        {
            "subject": f"{c} - Interview Request",
            "body": f"Hello {n},\n\nThank you for applying to {c} for our {r} role. We've reviewed your qualifications and would like to set up a call to discuss the opportunity further.\n\nWould you be available for a brief introductory call this week? The conversation would last about 20-30 minutes.\n\nPlease let me know what times work best for you.\n\nThanks,\n{rec}\n{c}",
        },
        {
            "subject": f"Moving forward with your application - {c}",
            "body": f"Hi {n},\n\nI'm excited to let you know that we'd like to advance your candidacy for the {r} role at {c}.\n\nThe next step in our process is a technical interview. This will be a 60-minute session focused on problem-solving and domain knowledge relevant to the position.\n\nCould you share your availability for the next two weeks? We'll work to find a time that suits you.\n\nBest,\n{rec}\n{c} Hiring Team",
        },
        {
            "subject": f"Interview details for {r} - {c}",
            "body": f"Dear {n},\n\nFollowing our review of your application, we are pleased to invite you to an on-site interview at {c} for the {r} position.\n\nDate: {date_str}\nLocation: {c} Headquarters\nTime: {time_str}\n\nYour interview day will include:\n- Meeting with the hiring manager\n- Technical assessment\n- Team meet and greet\n- Lunch with the team\n\nPlease bring a valid photo ID. We will reimburse any travel expenses.\n\nLooking forward to meeting you in person!\n\n{rec}\n{c}",
        },
        {
            "subject": f"Screening call - {r} at {c}",
            "body": f"Hi {n},\n\nThanks for your application to the {r} position at {c}. Your profile caught our attention and we'd love to learn more about you.\n\nI'd like to schedule a quick screening call to discuss your background and the opportunity. This would be a casual 20-minute conversation.\n\nAre you free sometime this week?\n\nCheers,\n{rec}\nTalent Acquisition, {c}",
        },
        {
            "subject": f"You've been selected! Next steps for {r} at {c}",
            "body": f"Dear {n},\n\nWe have reviewed your application for the {r} role at {c} and we are impressed with your background.\n\nWe would like to move you to the next step of our interview process. A recruiter will be reaching out to you within the next 2-3 business days to schedule an initial video interview.\n\nIn the meantime, please ensure your contact information is up to date in our system.\n\nBest regards,\nThe {c} Recruitment Team",
        },
    ]

    t = random.choice(templates)
    return c, t["subject"], t["body"], "interview"

#Randomly generated emails under "ACTION_REQUIRED"
def gen_action_required():
    c = rand_company()
    r = rand_role()
    n = rand_name()
    deadline = rand_date()
    deadline_str = deadline.strftime("%B %d, %Y")

    templates = [
        {
            "subject": f"Action Required: Complete your assessment for {c}",
            "body": f"Dear {n},\n\nThank you for applying to the {r} position at {c}. As the next step in our hiring process, we'd like you to complete an online assessment.\n\nPlease complete the assessment within 5 business days (by {deadline_str}). The assessment will take approximately 60-90 minutes.\n\nClick the link below to begin:\n[Assessment Link]\n\nIf you have any technical difficulties, please contact our support team.\n\nGood luck!\n{c} Recruiting",
        },
        {
            "subject": f"Complete your {c} application - Assessment Required",
            "body": f"Hi {n},\n\nCongratulations on being selected to move forward in the hiring process for {r} at {c}!\n\nTo continue, please complete the following online assessment:\n\nAssessment: Technical Skills Evaluation\nPlatform: HackerRank\nTime limit: 90 minutes\nDeadline: {deadline_str}\n\nPlease ensure you have a stable internet connection and a quiet environment.\n\nBest of luck,\n{c} Talent Team",
        },
        {
            "subject": f"Next Steps: {r} position at {c}",
            "body": f"Dear {n},\n\nThank you for your interest in the {r} role at {c}. We were pleased with your initial application and would like to invite you to complete the next step in our process.\n\nPlease complete the following:\n1. Online coding challenge (via CoderPad)\n2. Short video introduction (2-3 minutes)\n\nBoth must be completed by {deadline_str}.\n\nAccess the assessment portal here: [Link]\n\nReach out if you need accommodations.\n\n{c} Hiring Team",
        },
        {
            "subject": f"Assessment invitation from {c}",
            "body": f"Hi {n},\n\nAs part of the application process for {r} at {c}, we invite you to take our online assessment.\n\nThe assessment evaluates your analytical and problem-solving abilities. It should take about 45 minutes to complete.\n\nPlease complete it by {deadline_str}.\n\n[Start Assessment]\n\nNote: Once you begin, the timer will start and cannot be paused.\n\nThank you,\n{c}",
        },
        {
            "subject": f"Action Needed: {r} application at {c}",
            "body": f"Hello {n},\n\nWe are excited about your application for the {r} position at {c}. Before we can proceed, we need you to complete a few items:\n\n- Complete the technical assessment (estimated time: 60 minutes)\n- Upload a copy of your most recent transcript\n- Fill out the additional questionnaire\n\nPlease complete these steps by {deadline_str} to remain in consideration.\n\nThank you for your time and interest!\n\n{c} Recruiting Team",
        },
        {
            "subject": f"Complete Assessment - {c} Application",
            "body": f"Dear {n},\n\nYou have been invited to complete an assessment for the {r} position at {c}.\n\nAssessment Type: Situational Judgment Test\nDuration: Approximately 30 minutes\nDeadline: {deadline_str}\n\nThis assessment helps us understand how you approach workplace scenarios. There are no right or wrong answers.\n\nClick here to begin: [Assessment Link]\n\nBest regards,\n{c} People & Culture",
        },
        {
            "subject": f"HireVue Interview - {r} at {c}",
            "body": f"Hi {n},\n\nThank you for applying to {c}! As part of our selection process for the {r} role, we'd like you to complete a HireVue video interview.\n\nYou'll be asked to respond to a series of pre-recorded questions on camera. You can complete this at any time before {deadline_str}.\n\nTips:\n- Find a quiet, well-lit space\n- Test your camera and microphone beforehand\n- You'll have time to prepare before each question\n\nStart your HireVue interview: [Link]\n\n{c} Talent Acquisition",
        },
        {
            "subject": f"Coding Challenge: {r} - {c}",
            "body": f"{n},\n\nWe're impressed with your profile and would like to see your coding skills in action!\n\nPlease complete the following coding challenge for the {r} role at {c}:\n\nPlatform: CodeSignal\nDuration: 70 minutes\nLanguages: Python, Java, or C++\nDeadline: {deadline_str}\n\nThe challenge will test data structures, algorithms, and problem-solving skills.\n\nGood luck!\n{c} Engineering Recruiting",
        },
        {
            "subject": f"Please complete your application - {c}",
            "body": f"Dear {n},\n\nYour application for {r} at {c} is incomplete. To be considered for the role, please complete the following:\n\n- Finish the online application form\n- Submit your portfolio or work samples\n- Complete the pre-employment questionnaire\n\nDeadline: {deadline_str}\n\nLog in to our careers portal to complete these items.\n\nThank you,\n{c} HR",
        },
        {
            "subject": f"Background Check Required - {c}",
            "body": f"Hi {n},\n\nAs part of the hiring process for the {r} position at {c}, we need you to complete a background check authorization.\n\nPlease click the link below to provide the required information:\n[Background Check Portal]\n\nThis must be completed by {deadline_str}. The background check typically takes 3-5 business days to process.\n\nIf you have questions, contact us at hr@{c.lower().replace(' ', '')}.com.\n\nThank you,\n{c} Human Resources",
        },
    ]

    t = random.choice(templates)
    return c, t["subject"], t["body"], "action_required"

#Randomly generated emails under "ACCEPTANCE"
def gen_acceptance():
    c = rand_company()
    r = rand_role()
    n = rand_name()
    rec = rand_recruiter()
    start = rand_date() + timedelta(days=random.randint(14, 45))
    start_str = start.strftime("%B %d, %Y")
    salary = random.choice(["$85,000", "$95,000", "$105,000", "$115,000", "$125,000", "$140,000", "$155,000"])

    templates = [
        {
            "subject": f"Congratulations! Offer for {r} at {c}",
            "body": f"Dear {n},\n\nOn behalf of {c}, I am pleased to extend an offer of employment for the position of {r}.\n\nWe were truly impressed by your qualifications and believe you will be a great addition to our team. Please find the details of your offer below:\n\nPosition: {r}\nStart Date: {start_str}\nBase Salary: {salary} per year\nLocation: Hybrid\n\nPlease review the attached offer letter and respond within 5 business days.\n\nWelcome aboard!\n\n{rec}\nHR Director, {c}",
        },
        {
            "subject": f"Offer Letter - {r} at {c}",
            "body": f"Dear {n},\n\nCongratulations! We are excited to extend you an offer to join {c} as a {r}.\n\nYour start date is {start_str}. The compensation package includes:\n- Base salary: {salary}\n- Annual bonus target\n- Full benefits package\n- Stock options\n\nPlease sign and return the attached offer letter by the indicated deadline. If you have any questions about the offer or the onboarding process, don't hesitate to reach out.\n\nWe can't wait to have you on the team!\n\nBest,\n{rec}\n{c} People Team",
        },
        {
            "subject": f"Welcome to {c}!",
            "body": f"Dear {n},\n\nWe are thrilled to welcome you to the {c} family! Your offer for the {r} position has been accepted, and we're excited to have you join us.\n\nYour onboarding details:\n- Start date: {start_str}\n- Orientation: First day at 9:00 AM\n- Location: Main campus, Building A\n\nBefore your start date, please complete the following:\n1. Sign all onboarding documents\n2. Set up your benefits enrollment\n3. Complete the I-9 verification\n\nWe look forward to seeing you on {start_str}!\n\n{rec}\nOnboarding Coordinator, {c}",
        },
        {
            "subject": f"{c} - Your Offer of Employment",
            "body": f"Hi {n},\n\nI'm delighted to inform you that you have been selected for the {r} role at {c}. Congratulations on your successful interview process!\n\nThe formal offer letter is attached to this email. Key highlights:\n- Role: {r}\n- Start date: {start_str}\n- Compensation: {salary} base + bonus\n\nPlease review carefully and let me know if you have any questions. We'd appreciate your decision within one week.\n\nWelcome to the team!\n\n{rec}\n{c}",
        },
        {
            "subject": f"Great news from {c}!",
            "body": f"Dear {n},\n\nI am pleased to offer you the position of {r} at {c}. After meeting with our interview panel, everyone was impressed with your skills and experience, and we believe you'll make a significant contribution to our team.\n\nPlease review the attached offer details and compensation package. Your proposed start date is {start_str}.\n\nTo accept, please sign the offer letter and return it to us. We're excited to have you!\n\nCongratulations,\n{rec}\nVP of Talent, {c}",
        },
        {
            "subject": f"Job Offer: {r} - {c}",
            "body": f"{n},\n\nCongratulations on your offer! After a competitive selection process, we are pleased to extend an offer for the {r} position at {c}.\n\nOffer details:\n- Annual base: {salary}\n- Benefits effective from start date\n- Start date: {start_str}\n\nPlease accept or decline by responding to this email within 5 business days.\n\nLooking forward to your response!\n\n{c} Recruiting",
        },
        {
            "subject": f"Congratulations, {n}!",
            "body": f"Dear {n},\n\nIt gives me great pleasure to inform you that you have been selected as our new {r}. Congratulations on this well-deserved achievement!\n\nWe were thoroughly impressed throughout the interview process and are confident that you will thrive at {c}.\n\nYour offer letter with full details is attached. Please don't hesitate to reach out with any questions.\n\nWelcome aboard!\n\nBest regards,\n{rec}\nChief People Officer\n{c}",
        },
        {
            "subject": f"Your {c} offer",
            "body": f"Hi {n},\n\nFantastic news - we'd love to have you join us at {c}! I'm pleased to extend a formal offer for the {r} position.\n\nThe details:\n- Compensation: {salary} annually\n- Start date: {start_str}\n- Reports to: {rand_recruiter()}\n\nWe've attached the full offer package for your review. Take your time looking through it and let me know if you have any questions.\n\nWe're really excited about this!\n\n{rec}\n{c}",
        },
    ]

    t = random.choice(templates)
    return c, t["subject"], t["body"], "acceptance"

#Randomly generated emails under "UNRELATED"
def gen_unrelated():
    c = rand_company()
    n = rand_name()

    templates = [
        {
            "subject": f"New jobs that match your profile",
            "body": f"Hi {n},\n\nWe found new jobs that match your profile:\n\n- Data Analyst at {rand_company()}\n- Business Analyst at {rand_company()}\n- Software Engineer at {rand_company()}\n\nApply now on our platform!\n\nIf you wish to unsubscribe from these alerts, click here.\n\nJobBoard Team",
        },
        {
            "subject": f"Welcome to {c} Careers Portal",
            "body": f"Hi {n},\n\nWelcome to the {c} Careers Portal! Your account has been created successfully.\n\nHere's what you can do:\n- Browse open positions\n- Set up job alerts\n- Track your applications\n- Update your profile\n\nUsername: {n.lower().replace(' ', '.')}\n\nPlease verify your email address by clicking the link below.\n\n{c} Careers",
        },
        {
            "subject": f"Your weekly job digest",
            "body": f"Hi {n},\n\nHere's your weekly roundup of jobs based on your preferences:\n\n{rand_company()} - {rand_role()}\n{rand_company()} - {rand_role()}\n{rand_company()} - {rand_role()}\n{rand_company()} - {rand_role()}\n\nSee all recommendations on your dashboard.\n\nTo manage your notification preferences or unsubscribe, visit settings.\n\nBest,\nThe Jobs Team",
        },
        {
            "subject": f"Verify your email address",
            "body": f"Hi {n},\n\nPlease confirm your email address to complete your registration on {c}'s career portal.\n\nClick the button below to verify:\n[Verify Email]\n\nThis link will expire in 24 hours.\n\nIf you did not create an account, please ignore this email.\n\n{c}",
        },
        {
            "subject": f"Complete your profile on {c}",
            "body": f"Hi {n},\n\nYour profile on {c}'s talent network is only 60% complete. A complete profile increases your chances of being found by recruiters.\n\nAdd these to stand out:\n- Work experience details\n- Skills and certifications\n- Portfolio links\n- Profile photo\n\nUpdate your profile now: [Link]\n\n{c} Talent Community",
        },
        {
            "subject": f"Job alert: {rand_role()} positions near you",
            "body": f"Hi {n},\n\nNew jobs matching your saved search:\n\n{rand_role()} - {rand_company()} (Remote)\n{rand_role()} - {rand_company()} (Hybrid)\n{rand_role()} - {rand_company()} (On-site)\n\nDon't miss out - apply today!\n\nManage alerts | Unsubscribe\n\nLinkedIn Jobs",
        },
        {
            "subject": f"Password reset request",
            "body": f"Hi {n},\n\nWe received a request to reset your password for your {c} careers account.\n\nClick here to reset your password: [Reset Link]\n\nThis link expires in 1 hour. If you didn't request this, please ignore this email.\n\nSecurity Team\n{c}",
        },
        {
            "subject": f"Tips for your job search",
            "body": f"Hi {n},\n\nLooking for your next opportunity? Here are this week's tips:\n\n1. Tailor your resume for each application\n2. Follow up within a week of applying\n3. Practice common interview questions\n4. Network with industry professionals\n\nExplore open roles: [Browse Jobs]\n\nGood luck!\nCareer Advice Team",
        },
        {
            "subject": f"{c} Talent Community Newsletter",
            "body": f"Hello {n},\n\nWelcome to the {c} Talent Community monthly newsletter!\n\nThis month:\n- Inside look: Life at {c}\n- Employee spotlight: Meet our team\n- Upcoming hiring events\n- New blog: Career development tips\n\nStay connected and be the first to know about new opportunities.\n\nUnsubscribe from this newsletter\n\n{c} Employer Branding Team",
        },
        {
            "subject": f"Similar jobs you might like",
            "body": f"{n}, based on your recent activity, we think you'll be interested in these roles:\n\n{rand_role()} at {rand_company()} - Posted 2 days ago\n{rand_role()} at {rand_company()} - Posted 3 days ago\n{rand_role()} at {rand_company()} - Posted 1 week ago\n\nApply with one click using your saved profile.\n\nJob recommendations are based on your search history. Unsubscribe here.",
        },
    ]

    t = random.choice(templates)
    return c, t["subject"], t["body"], "unrelated"


#Working with the dataset

#function generates the base of 2000 emails, 
# with the following amount in each categpry:
def generate_dataset(
    n_acceptance=300,
    n_rejection=400,
    n_interview=350,
    n_action=350,
    n_in_process=400,
    n_unrelated=200,
):
    #generating using functions above
    generators = [
        (gen_acceptance, n_acceptance),
        (gen_rejection, n_rejection),
        (gen_interview, n_interview),
        (gen_action_required, n_action),
        (gen_in_process, n_in_process),
        (gen_unrelated, n_unrelated),
    ]

    rows = []
    idx = 0
    for gen_fn, count in generators:
        for _ in range(count):
            company, subject, body, label = gen_fn()
            d = rand_date()
            rows.append({
                "Unnamed: 0": idx,
                "sender": rand_sender(company),
                "date": d.strftime("%-m/%-d/%y, %-I:%M %p"),
                "parsed_datetime": d.strftime("%Y-%m-%d %H:%M:%S"),
                "date_only": d.strftime("%Y-%m-%d"),
                "week": d.isocalendar()[1],
                "month": d.month,
                "year": d.year,
                "days_since": (datetime(2025, 4, 11) - d).days,
                "email_body": body,
                "company": company,
                "subject": subject,
                "true_label": label,  # GT for training
            })
            idx += 1

    random.shuffle(rows)
    return rows


def main():
    print("***Generating dataset***")
    print()

    rows = generate_dataset()
    total = len(rows)

    # Count by label
    from collections import Counter
    label_counts = Counter(r["true_label"] for r in rows)
    print(f"Total emails: {total}")
    print("Class distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:20s} {count:4d} ({pct:5.1f}%) {bar}")

    # Save
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_emails.csv")

    fieldnames = [
        "Unnamed: 0", "sender", "date", "parsed_datetime", "date_only",
        "week", "month", "year", "days_since", "email_body",
        "company", "subject", "true_label",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
