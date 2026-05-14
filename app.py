from flask import Flask, render_template, request, redirect, url_for
from email_parser import classify_email
from reply_generator import generate_reply, check_groq_configured
from calendar_helper import create_calendar_event

app = Flask(__name__)
app.secret_key = "replace-with-a-random-string-abc123"


# ── Route 1: Home — show the input form ──────────────────────
@app.route("/")
def index():
    groq_ok = check_groq_configured()  # warn user if API key is missing
    return render_template("index.html", groq_ok=groq_ok)


# ── Route 2: Analyze email (POST form submission) ─────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    email_text = request.form.get("email_text", "").strip()
    tone       = request.form.get("tone", "professional")

    if not email_text:
        return redirect(url_for("index"))

    # Step 1: classify intent + extract entities
    result     = classify_email(email_text)
    intent     = result["intent"]
    confidence = result["confidence"]
    dates      = result["dates"]
    times      = result["times"]

    # Step 2: generate AI reply using Groq
    reply = generate_reply(email_text, intent, tone=tone)

    # Step 3: pass everything to the result template
    return render_template(
        "result.html",
        intent=intent,
        confidence=int(confidence * 100),
        dates=dates,
        times=times,
        reply=reply,
        email_text=email_text,
        tone=tone,
    )


# ── Route 3: Schedule Google Calendar event ───────────────────
@app.route("/schedule", methods=["POST"])
def schedule():
    title = request.form.get("title", "Meeting")
    date  = request.form.get("date", "")
    time  = request.form.get("time", "10:00")

    try:
        event     = create_calendar_event(title, date, time)
        event_url = event.get("htmlLink", "#")
        return render_template(
            "result.html", scheduled=True, event_url=event_url
        )
    except Exception as e:
        return render_template(
            "result.html", scheduled=False,
            error=f"Calendar error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)