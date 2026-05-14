import spacy
import pandas as pd
import os
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

print("Loading real datasets...")

# ══════════════════════════════════════════════════════════════
# DATASET 1: UCI SMS Spam — fix column name issue
# The dataset columns are "sms" and "label", not "label" and "text"
# ══════════════════════════════════════════════════════════════
try:
    spam_ds = load_dataset("ucirvine/sms_spam", split="train")
    spam_df = pd.DataFrame(spam_ds)
    print(f"  SMS Spam columns: {list(spam_df.columns)}")  # debug
    # Rename whichever column has the text
    if "sms" in spam_df.columns:
        spam_df = spam_df.rename(columns={"sms": "text"})
    elif "message" in spam_df.columns:
        spam_df = spam_df.rename(columns={"message": "text"})
    # label column: 0=ham, 1=spam OR "ham"/"spam"
    spam_df["label"] = spam_df["label"].astype(str).map({
        "ham": "query", "spam": "spam",
        "0":   "query", "1":    "spam",
    })
    spam_df = spam_df[["text", "label"]].dropna()
    print(f"  UCI SMS Spam:          {len(spam_df):,} samples")
except Exception as e:
    print(f"  SMS Spam failed: {e}")
    spam_df = pd.DataFrame(columns=["text", "label"])

# ══════════════════════════════════════════════════════════════
# DATASET 2: Enron emails (aeslc) → "query"
# ══════════════════════════════════════════════════════════════
try:
    enron_ds = load_dataset("aeslc", split="train")
    enron_df = pd.DataFrame(enron_ds)[["email_body"]].rename(
        columns={"email_body": "text"}
    )
    enron_df["label"] = "query"
    enron_df = enron_df[enron_df["text"].str.len() > 30]
    enron_df = enron_df.sample(1500, random_state=42)
    print(f"  Enron (aeslc):         {len(enron_df):,} samples")
except Exception as e:
    print(f"  Enron skipped: {e}")
    enron_df = pd.DataFrame(columns=["text", "label"])

# ══════════════════════════════════════════════════════════════
# DATASET 3: SetFit/enron_spam filtered for meeting keywords
# ══════════════════════════════════════════════════════════════
try:
    meeting_ds = load_dataset("SetFit/enron_spam", split="train")
    meeting_df = pd.DataFrame(meeting_ds)
    # Strict scheduling keywords only — reduces false positives
    pattern = (r"(?i)\b(schedule a|set up a|arrange a|book a|"
               r"meet on|meet at|meeting at|call at|sync at|"
               r"are you free|are you available|your availability|"
               r"would you be available|let.s meet|let.s connect|"
               r"conference call|zoom call|teams call|video call)\b")
    meeting_df = meeting_df[
        meeting_df["text"].str.contains(pattern, regex=True, na=False)
    ][["text"]].copy()
    meeting_df["label"] = "meeting"
    meeting_df = meeting_df.sample(min(600, len(meeting_df)), random_state=42)
    print(f"  Meeting (Enron filter):{len(meeting_df):,} samples")
except Exception as e:
    print(f"  Meeting dataset skipped: {e}")
    meeting_df = pd.DataFrame(columns=["text", "label"])

# ══════════════════════════════════════════════════════════════
# DATASET 4: Spam — curated real spam patterns
# Needed because SMS spam (if missing) leaves no spam class
# ══════════════════════════════════════════════════════════════
spam_curated_texts = [
    "Congratulations! You have won a $1,000,000 prize. Click here to claim.",
    "You have been selected for a FREE iPhone 15. Claim now before it expires!",
    "Limited time offer: make $5000 a day working from home. No experience needed.",
    "Your account has been compromised. Verify your details immediately here.",
    "FREE gift waiting for you — click to claim before it expires tonight.",
    "Hot singles in your area want to meet you tonight. Join free now.",
    "You are our lucky winner! Collect your prize immediately. Act fast.",
    "Buy cheap medications online — no prescription needed. Discreet shipping.",
    "Earn money fast — guaranteed investment returns of 500%. Risk free.",
    "Your PayPal account is suspended. Click to reactivate immediately now.",
    "Lose 30 pounds in 30 days with this one weird trick doctors hate.",
    "Nigerian prince needs your help transferring $10 million. Confidential.",
    "Exclusive deal: replica luxury watches at 90% off. Limited stock.",
    "You have unclaimed lottery winnings — verify your identity today.",
    "Work from home and make thousands per week, 100% guaranteed income.",
    "Click here to see who viewed your profile recently. Free access.",
    "FREE iPhone — just pay shipping. Limited time offer ends tonight.",
    "Your bank account has been locked. Confirm your PIN to restore access.",
    "Double your bitcoin in 24 hours — guaranteed returns, no risk.",
    "Urgent: your computer has a virus — call this toll-free number now.",
    "Win a holiday trip to Maldives — you have been randomly selected.",
    "Make money online doing surveys — earn $500 per day from home.",
    "Your inheritance of $2.5 million is waiting to be claimed. Contact us.",
    "Special offer on medications — buy online without prescription today.",
    "Claim your exclusive membership reward before it expires at midnight.",
    "Investment opportunity of a lifetime — guaranteed profits every month.",
    "You are pre-approved for a loan with no credit check required.",
    "Shocking celebrity secret revealed — click to see more now.",
    "FREE diet pills — lose weight without any exercise or diet changes.",
    "Urgent: update your account information immediately to avoid suspension.",
    "Winner! Your email address won our weekly draw. Claim your $500 now.",
    "Multi-level marketing opportunity — earn passive income from home today.",
    "Your subscription has expired. Renew now or lose all your data forever.",
    "Exclusive crypto investment — guaranteed 300% returns in 7 days.",
    "You have a pending package. Pay $2 customs fee to release it now.",
    "Romance opportunity — beautiful singles waiting to chat with you.",
    "Congratulations, you are today's lucky visitor. Claim your reward.",
    "WARNING: your antivirus has expired. Download protection now. Free.",
    "Make $3000 per week stuffing envelopes at home. Apply immediately.",
    "Final notice: your account will be deleted unless you verify today.",
]
spam_curated_df = pd.DataFrame({
    "text": spam_curated_texts,
    "label": ["spam"] * len(spam_curated_texts)
})

# ══════════════════════════════════════════════════════════════
# DATASET 5: Urgent — curated real-pattern examples
# ══════════════════════════════════════════════════════════════
urgent_texts = [
    "URGENT: Our production server is completely down. All customers locked out. Fix immediately.",
    "Critical alert: Payment processing system failed. All transactions being declined right now.",
    "Emergency: Data breach detected on customer database. Immediate action required.",
    "ASAP: Client demo in 45 minutes and application is throwing 500 errors. Help needed.",
    "High priority: Entire engineering team locked out of GitHub. Deployment deadline today.",
    "URGENT: Wrong invoice sent to 500 clients. Please recall and resend immediately.",
    "Critical: AWS bill spiked 10x overnight. Possible cryptomining attack on our servers.",
    "SOS: CEO needs board presentation in 20 minutes. File is missing from shared drive.",
    "Immediate attention: Legal has flagged product for compliance violation. Respond today.",
    "URGENT: Hospital patient management system offline. Staff cannot access patient records.",
    "Time-sensitive: Regulatory submission due in 2 hours. Still missing 3 key signatures.",
    "Red alert: All backup systems failing simultaneously. Data loss risk is imminent now.",
    "CRITICAL: Payroll system error — employees did not receive their salaries today.",
    "Urgent escalation: Key client threatening legal action if not resolved by 5pm today.",
    "Emergency: Office flooded. Need all staff to work remotely with immediate effect.",
    "URGENT: Security certificate expired. Users seeing browser warnings across entire site.",
    "Priority one: Warehouse management system crash has halted all shipping operations.",
    "Immediate fix needed: API rate limit exceeded, all third-party integrations are broken.",
    "ASAP escalation: Wrong product shipped to 200 customers. Return process needed now.",
    "Critical system failure: Entire customer support team cannot log in to helpdesk.",
    "URGENT: Press release going out in 30 minutes with completely incorrect pricing.",
    "Emergency: Three servers overheating in datacenter. Automatic shutdown in 10 minutes.",
    "High priority alert: Social media account hacked and posting offensive content now.",
    "URGENT: Board meeting starts in 1 hour and entire conferencing system is down.",
    "Critical: Competitor has published our unreleased product roadmap. PR crisis unfolding.",
    "Immediate response: Health and safety incident reported on factory floor right now.",
    "URGENT: System audit tomorrow and all compliance reports are completely missing.",
    "Time-critical: Flight booking for CEO delegation expires in 45 minutes. Book now.",
    "Emergency response: Customer data accidentally made public on our website. Fix now.",
    "URGENT: All microservices returning null responses after the latest deployment.",
    "Need urgent help: entire website is returning 404 for all pages since this morning.",
    "CRITICAL ISSUE: customer credit card data may have been exposed. Investigate NOW.",
    "Escalating urgently: three enterprise clients reporting complete service outage.",
    "Emergency maintenance needed immediately: database disk at 99% capacity and growing.",
    "URGENT please respond: signed contract must be returned in the next 30 minutes.",
]
urgent_df = pd.DataFrame({
    "text": urgent_texts,
    "label": ["urgent"] * len(urgent_texts)
})
print(f"  Spam (curated):        {len(spam_curated_df):,} samples")
print(f"  Urgent (curated):      {len(urgent_df):,} samples")

# ══════════════════════════════════════════════════════════════
# COMBINE ALL
# ══════════════════════════════════════════════════════════════
df = pd.concat(
    [spam_df, enron_df, meeting_df, spam_curated_df, urgent_df],
    ignore_index=True
)
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 10]

print(f"\nTotal: {len(df):,} training samples")
print(df["label"].value_counts().to_string())

# ══════════════════════════════════════════════════════════════
# TRAIN: TF-IDF + LinearSVM
# ══════════════════════════════════════════════════════════════
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), max_features=25000,
    sublinear_tf=True, min_df=1,
)

# Use class_weight to handle imbalance — urgent has fewer samples
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight("balanced", df["label"])

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    df["text"], df["label"], sample_weights,
    test_size=0.2, random_state=42, stratify=df["label"]
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# class_weight="balanced" inside SVM also helps with urgent/spam imbalance
svm = LinearSVC(max_iter=3000, C=1.0, class_weight="balanced")
classifier = CalibratedClassifierCV(svm, cv=3)
classifier.fit(X_train_vec, y_train)

y_pred = classifier.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ══════════════════════════════════════════════════════════════
# MAIN FUNCTION — called by app.py
# ══════════════════════════════════════════════════════════════
def classify_email(email_text: str) -> dict:
    doc    = nlp(email_text)
    dates  = [e.text for e in doc.ents if e.label_ == "DATE"]
    times  = [e.text for e in doc.ents if e.label_ == "TIME"]
    people = [e.text for e in doc.ents if e.label_ == "PERSON"]
    vec        = vectorizer.transform([email_text])
    intent     = classifier.predict(vec)[0]
    confidence = round(float(classifier.predict_proba(vec).max()), 2)
    return {
        "intent": str(intent), "confidence": confidence,
        "dates": dates, "times": times, "people": people,
    }


# ══════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        ("Can we schedule a call next Tuesday at 3pm?",         "meeting"),
        ("What is the status of my refund from last week?",     "query"),
        ("URGENT: server is completely down, need help NOW!",   "urgent"),
        ("Congratulations! You won $1,000,000. Click here.",    "spam"),
        ("Are you free for a quick sync tomorrow morning?",     "meeting"),
        ("I have a question about the invoice you sent me.",    "query"),
        ("Would you be available for a call next Monday?",      "meeting"),
        ("CRITICAL: payment gateway down, all orders failing.", "urgent"),
        ("Free iPhone winner selected! Claim your prize now.",  "spam"),
        ("Could you clarify the terms in the contract?",        "query"),
    ]
    print("\n── Live Test ─────────────────────────────────────────")
    correct = 0
    for text, expected in tests:
        r = classify_email(text)
        ok = "✓" if r["intent"] == expected else "✗"
        if r["intent"] == expected:
            correct += 1
        print(f"  {ok} [{r['intent'].upper():8}] {r['confidence']*100:.0f}%  →  {text[:58]}")
    print(f"\n  Score: {correct}/{len(tests)} correct") 