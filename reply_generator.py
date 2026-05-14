# pip install groq
from groq import Groq
import os

# Get your free API key at console.groq.com
# Then run in PowerShell: $env:GROQ_API_KEY="gsk_your_key_here"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Tone descriptions injected into the prompt
TONES = {
    "professional": "formal and professional, no contractions",
    "friendly":     "warm and approachable, uses contractions naturally",
    "concise":      "ultra-brief, under 50 words, get straight to the point",
    "apologetic":   "empathetic and apologetic in tone",
    "assertive":    "confident and direct, no hedging language",
}

# Intent-specific instructions injected into the prompt
INTENT_INSTRUCTIONS = {
    "meeting": "Acknowledge the meeting request. Confirm availability or ask to clarify date/time.",
    "query":   "Acknowledge the question. Say you will respond within 24 hours if you need time.",
    "urgent":  "Start with: I have received your urgent message and am addressing it immediately.",
    "spam":    "",
}


def generate_reply(email_text: str, intent: str,
                   tone: str = "professional") -> str:
    """
    Generates an AI email reply using Groq (free, cloud-based, no GPU needed).

    Args:
        email_text : the original email to reply to
        intent     : classified intent from email_parser (meeting/query/urgent/spam)
        tone       : reply tone — professional / friendly / concise / apologetic / assertive
    Returns:
        Reply string ready to display in the UI
    """
    if intent == "spam":
        return "Spam detected — no reply recommended."

    tone_desc   = TONES.get(tone, TONES["professional"])
    intent_inst = INTENT_INSTRUCTIONS.get(intent, "")

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",   # free Llama 3 model on Groq — fast & reliable
            max_tokens=300,
            temperature=0.7,          # 0 = robotic, 1 = creative, 0.7 = balanced
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a professional email assistant.
Write a reply to the email the user provides. Follow these rules exactly:
1. Tone: {tone_desc}
2. {intent_inst}
3. Output the reply body ONLY — no subject line, no preamble like "Here is your reply:"
4. Never invent facts, names, or dates not in the original email
5. End with: Best regards, [Your Name]
6. Maximum 120 words"""
                },
                {
                    "role": "user",
                    "content": f"Write a reply to this email:\n\n{email_text}"
                }
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Graceful fallback if API key is missing or Groq is unreachable
        print(f"Groq error: {e}")
        return (
            "Thank you for your email. I have received your message "
            "and will respond shortly.\n\nBest regards,\n[Your Name]"
        )


def check_groq_configured() -> bool:
    """Returns True if GROQ_API_KEY environment variable is set."""
    return bool(os.environ.get("GROQ_API_KEY"))