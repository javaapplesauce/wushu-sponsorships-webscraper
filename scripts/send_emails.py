import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()  # load EMAIL_USER, EMAIL_PASS from .env

def send_email(to_email, subject, body, sender_email, sender_pass):
    """Send a single email via Gmail SMTP."""
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_pass)
            server.send_message(msg)
            print(f"✅ Sent email to {to_email}")
    except Exception as e:
        print(f"⚠️  Failed to send email to {to_email}: {e}")

def main():
    input_file = "personalized_emails/mail_merge.json"  # path to your JSON file

    # Load sender credentials
    sender_email = os.getenv("EMAIL_USER")
    sender_pass = os.getenv("EMAIL_PASS")

    if not sender_email or not sender_pass:
        print("❌ Missing EMAIL_USER or EMAIL_PASS in .env file.")
        return

    # Load the personalized emails
    with open(input_file, "r", encoding="utf-8") as f:
        emails = json.load(f)

    print(f"[INFO] Loaded {len(emails)} emails from {input_file}")

    # Send each email
    for i, entry in enumerate(emails):
        to_email = entry.get("to_email")
        subject = entry.get("subject")
        body = entry.get("body")

        if not to_email:
            print(f"[{i+1}] ⚠️ Skipping row with no recipient.")
            continue

        print(f"[{i+1}] ✉️  Sending to: {to_email} ({subject})")
        send_email(to_email, subject, body, sender_email, sender_pass)

if __name__ == "__main__":
    main()
