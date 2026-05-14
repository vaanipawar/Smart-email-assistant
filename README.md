# Smart Email Assistant

A Flask web app that classifies incoming emails by intent, generates AI-powered replies, and auto-schedules meetings in Google Calendar — all running locally.

---

## Features

- **Intent Classification** — Classifies emails into `meeting`, `query`, `urgent`, or `spam` using a TF-IDF + LinearSVM model trained on ~6,500 real-world emails
- **Named Entity Extraction** — Uses spaCy to extract dates, times, and people from email text
- **AI Reply Generation** — Generates contextual email replies via Groq (free cloud API, Llama 3 model) with 5 selectable tones
- **Google Calendar Integration** — One-click event creation for meeting emails using the Google Calendar API
- **Spam Filtering** — Detects spam and suppresses reply generation automatically

---

## Project Structure

```
smart-email-assistant/
│
├── app.py                  # Flask app — routes and orchestration
├── email_parser.py         # ML model training + classify_email() function
├── reply_generator.py      # Groq API integration for AI reply generation
├── calendar_helper.py      # Google Calendar OAuth + event creation
├── requirements.txt        # Python dependencies
│
└── templates/
    ├── index.html          # Email input form
    └── result.html         # Classification results + reply + scheduler
```

---

## How It Works

```
User pastes email
      │
      ▼
email_parser.py
  ├── spaCy NER  ──────────────► Extract dates, times, people
  └── TF-IDF + SVM ────────────► Predict intent + confidence
      │
      ▼
reply_generator.py
  └── Groq (Llama 3) ──────────► Generate reply in chosen tone
      │
      ▼
result.html
  └── If intent == "meeting" ──► calendar_helper.py ──► Google Calendar
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/vaanipawar/smart-email-assistant
cd smart-email-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set up Groq API key (free)

Get your free API key at [console.groq.com](https://console.groq.com), then set it as an environment variable:

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="gsk_your_key_here"
```

**macOS / Linux:**
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

> The app works without a Groq key — it falls back to a default reply template.

### 4. Set up Google Calendar (optional)

To enable one-click calendar scheduling:

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → Create a project
2. Enable the **Google Calendar API**
3. Create **OAuth 2.0 credentials** → Download as `credentials.json`
4. Place `credentials.json` in the project root

On first run, a browser window will open for Google login. After that, the token is cached in `token.pickle` automatically.

> Skip this step if you don't need calendar integration — all other features work without it.

### 5. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## ML Model Details

| Component | Detail |
|-----------|--------|
| Vectorizer | TF-IDF, unigrams + bigrams, 25,000 features |
| Classifier | LinearSVC with Platt scaling (CalibratedClassifierCV) |
| Class balancing | `class_weight="balanced"` + sample weights |
| Test accuracy | ~91% across 4 intent classes |

### Training Data Sources

| Dataset | Source | Label |
|---------|--------|-------|
| UCI SMS Spam | HuggingFace `ucirvine/sms_spam` | spam / query |
| Enron emails | HuggingFace `aeslc` | query |
| Enron meeting emails | HuggingFace `SetFit/enron_spam` | meeting |
| Curated spam patterns | Hand-written examples | spam |
| Curated urgent patterns | Hand-written examples | urgent |

---

## Reply Tones

| Tone | Description |
|------|-------------|
| Professional | Formal, no contractions |
| Friendly | Warm, approachable |
| Concise | Under 50 words |
| Apologetic | Empathetic in tone |
| Assertive | Direct, no hedging |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Optional | Groq API key for AI reply generation |

---

## Dependencies

Key packages (see `requirements.txt` for full list):

- `Flask` — web framework
- `scikit-learn` — TF-IDF vectorizer + LinearSVM classifier
- `spacy` + `en_core_web_sm` — named entity recognition
- `groq` — Groq API client for Llama 3 reply generation
- `google-api-python-client` + `google-auth-oauthlib` — Google Calendar integration
- `python-dateutil` — human-readable date parsing ("next Tuesday", "tomorrow")
- `datasets` — HuggingFace datasets for model training

---

## Limitations

- The model trains fresh on every startup (no saved model file) — takes ~10–20 seconds on first load
- Google Calendar integration requires a `credentials.json` file from Google Cloud Console
- Groq's free tier has rate limits; the app falls back gracefully if the API is unavailable
- The `urgent` class has fewer training samples than others, which may affect recall on edge cases
