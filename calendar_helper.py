import pickle, os
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Only request permission to create/edit events — nothing else
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def get_calendar_service():
    """
    Returns authenticated Google Calendar client.
    First run: opens browser for Google login.
    Subsequent runs: uses cached token.pickle silently.
    """
    creds = None

    # token.pickle stores your access token after first login
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)

    # Refresh expired token automatically if possible
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    # No valid credentials → start browser OAuth flow
    elif not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        # Opens browser, user logs in, token is returned
        creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as f:
            pickle.dump(creds, f)  # cache so login only happens once

    return build("calendar", "v3", credentials=creds)


def create_calendar_event(title: str, date_str: str,
                            time_str: str = "10:00",
                            duration_hours: int = 1) -> dict:
    """
    Creates a Google Calendar event and returns the event object.
    The returned dict contains htmlLink — a direct URL to the event.
    """
    service = get_calendar_service()

    # dateutil parses human strings: "next Tuesday", "tomorrow", "15 June"
    start_dt = dateparser.parse(f"{date_str} {time_str}")
    end_dt   = start_dt + timedelta(hours=duration_hours)

    event = {
        "summary": title,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": "Asia/Kolkata",  # change to your timezone
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
    }

    # calendarId="primary" = user's main Google Calendar
    created = service.events().insert(
        calendarId="primary", body=event
    ).execute()

    return created  # created["htmlLink"] = URL to the new event