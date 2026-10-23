from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
import json
import os
import urllib.parse

app = FastAPI(title="Safe Sight Guardian Emergency Backend")

# backend/server.py -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_FILE = os.path.join(BASE_DIR, "backend", "events.json")
VIDEO_FOLDER = os.path.join(BASE_DIR, "emergency_clips")
DASHBOARD_FOLDER = os.path.join(BASE_DIR, "dashboard")
PHASE2_FOLDER = os.path.join(BASE_DIR, "phase 2")
PHASE2_CANDIDATE_FOLDER = os.path.join(PHASE2_FOLDER, "candidate_clips")
PHASE2_BROWSER_FOLDER = os.path.join(PHASE2_CANDIDATE_FOLDER, "browser_ready")
PHASE2_EVENTS_FILE = os.path.join(PHASE2_CANDIDATE_FOLDER, "events.jsonl")
PHASE2_STATUS_FILE = os.path.join(PHASE2_CANDIDATE_FOLDER, "status.json")

for folder in (VIDEO_FOLDER, DASHBOARD_FOLDER, PHASE2_CANDIDATE_FOLDER, PHASE2_BROWSER_FOLDER):
    os.makedirs(folder, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=VIDEO_FOLDER), name="videos")
app.mount("/candidate-clips", StaticFiles(directory=PHASE2_CANDIDATE_FOLDER), name="candidate-clips")
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_FOLDER, html=True), name="dashboard")

class EmergencyRequest(BaseModel):
    video: str | None = None

def load_events():
    if not os.path.isfile(EVENTS_FILE):
        return []
    try:
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []

def load_phase2_events():
    events = []
    if not os.path.isfile(PHASE2_EVENTS_FILE):
        return events
    try:
        with open(PHASE2_EVENTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    filename = os.path.basename(str(event.get("clip_filename") or event.get("clip_path") or ""))
                    if filename and filename.lower().endswith(".mp4") and os.path.isfile(os.path.join(PHASE2_CANDIDATE_FOLDER, filename)):
                        event["clip_filename"] = filename
                        browser_filename = os.path.basename(str(event.get("browser_clip_filename") or filename))
                        browser_path = os.path.join(PHASE2_BROWSER_FOLDER, browser_filename)
                        if os.path.isfile(browser_path):
                            event["browser_clip_filename"] = browser_filename
                            event["clip_url"] = f"/candidate-clips/browser_ready/{urllib.parse.quote(browser_filename)}"
                        else:
                            event["clip_url"] = f"/candidate-clips/{urllib.parse.quote(filename)}"
                    else:
                        event["clip_url"] = None
                    event.setdefault("event_type", "PHASE2_CANDIDATE")
                    event.setdefault("review_status", "pending_human_review")
                    events.append(event)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return sorted(events, key=lambda e: str(e.get("timestamp", "")), reverse=True)

@app.get("/")
def home():
    return RedirectResponse(url="/dashboard/")

@app.get("/health")
def health():
    return {"status": "ok", "phase1_events": len(load_events()), "phase2_events": len(load_phase2_events())}

@app.post("/emergency")
def emergency_alert(data: EmergencyRequest):
    if os.getenv("SAFE_SIGHT_PUBLIC_READ_ONLY", "false").lower() == "true":
        raise HTTPException(status_code=403, detail="Emergency event submission is disabled in public demo mode.")
    events = load_events()
    alert = {
        "id": (max([int(e.get("id", 0)) for e in events] or [0]) + 1),
        "event": "EMERGENCY_DETECTED",
        "status": "CRITICAL",
        "camera": "CAMERA-01",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "video": os.path.basename(data.video) if data.video else None,
    }
    events.append(alert)
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    print("Safe Sight Guardian Phase 1 emergency received:", alert, flush=True)
    return alert

@app.get("/events")
def get_events():
    return load_events()

@app.get("/phase2-events")
def get_phase2_events():
    return load_phase2_events()

@app.get("/phase2-status")
def get_phase2_status():
    if not os.path.isfile(PHASE2_STATUS_FILE):
        return {"state": "STARTING", "message": "Waiting for Phase 2 worker", "max_score": 0, "mean_score": 0}
    try:
        with open(PHASE2_STATUS_FILE, "r", encoding="utf-8") as f:
            status = json.load(f)
        return status if isinstance(status, dict) else {"state": "UNKNOWN"}
    except (OSError, json.JSONDecodeError):
        return {"state": "UNKNOWN", "message": "Status file could not be read"}
