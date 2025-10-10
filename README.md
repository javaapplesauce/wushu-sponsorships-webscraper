## 🐉 Wushu Sponsorships Webscraper

Automates sponsorship outreach for Columbia Wushu. Conducts web scraping, then personalized email generation, then delivery.

This pipeline:

1. **Finds potential sponsors** (boba shops, Asian markets, skincare shops, etc.)
2. **Personalizes outreach emails** using a premade template and Google Gemini API calls
3. **Sends sponsorship requests automatically**

---

### 🧭 Directory Structure

```
wushu-sponsorships-webscraper/
├── scripts/
│   ├── webscraper.py              # Step 1: Scrape local shop data
│   ├── personalize_emails.py      # Step 2: Personalize emails via Gemini API
│   ├── send_emails.py             # Step 3: Send emails via Gmail SMTP
│   └── run_all.py                 # Master pipeline — runs all three steps in order
│
├── sponsorship_template.txt       # Base email template with <dynamic> sections
├── requirements.txt               # Dependencies
├── .env                           # Your private API keys and credentials
├── .gitignore                     
│
├── results_nyc/                   # Output folder for webscraper results
│   ├── shops.csv
│   └── shops.json
│
├── personalized_emails/           # Output folder for personalized emails
│   ├── mail_merge.csv
│   └── mail_merge.json
│
└── README.md                      # This documentation

```

---

### ⚙️ Setup

#### 1. Clone the repo

```bash
git clone https://github.com/javaapplesauce/wushu-sponsorships-webscraper.git
cd wushu-sponsorships-webscraper
```

#### 2. Create and activate your environment

```bash
conda create -n wushu python=3.10
conda activate wushu
pip install -r requirements.txt
```

#### 3. Add your credentials

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_gemini_api_key
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_specific_password
```

> 💡 *To use Gmail, create an “App Password” under*
> [Google Account → Security → App Passwords](https://myaccount.google.com/apppasswords)

---

### 🚀 Usage

#### **Option A — Full pipeline (recommended)**

Run everything automatically:

```bash
python run_all.py
```

This will:

1. Run the webscraper (find shops + contact info)
2. Generate personalized sponsorship emails
3. Send each email

#### **Option B — Step-by-step**

If you want to inspect intermediate results:

```bash
# 1️⃣ Scrape businesses
python webscraper.py --location "New York, NY" --radius 2000 --categories "boba,matcha,skincare"

# 2️⃣ Personalize emails
python personalize_emails.py \
  --csv results_nyc/shops.csv \
  --template sponsorship_template.txt \
  --event-location "Columbia University, NY" \
  --event-date "10/25" \
  --event-time "3:00 PM" \
  --event-place "Columbia Sundial" \
  --out personalized_emails

# 3️⃣ Send emails
python send_emails.py
```

---

### 🧠 Customization

| File                       | Purpose                                                                           |
| -------------------------- | --------------------------------------------------------------------------------- |
| `sponsorship_template.txt` | Base email body. You can add `<dynamic>...</dynamic>` sections for LLM rewriting. |
| `run_all.py`               | Controls sequence, event details, and scraping parameters.                        |
| `.env`                     | Securely stores API keys and credentials.                                         |
| `requirements.txt`         | Dependencies for reproducibility.                                                 |

---

### ⚡ Example Output

After a successful run:

```
results_nyc/shops.csv         → scraped shop info
personalized_emails/mail_merge.json  → final personalized emails
```

Console output:

```
[1/333] 🔄 Processing: Gong Cha (nyc@gongcha.com)
[1/333] ✅ Personalized email written
[1/333] ✉️ Sent email to nyc@gongcha.com
...
[SUMMARY] Successfully sent 333 emails.
```

---

### 🛡️ Safety Notes

* **Do not commit `.env` or generated email data.**
* **Use responsibly** — avoid spam.
  This project is for *student outreach automation*, not mass marketing.

---

### 💡 Future Ideas

* Integrate with Google Sheets for live sponsor tracking
* Add progress dashboards
* Use the Gmail API (for higher send limits)
* Add support for other event templates
