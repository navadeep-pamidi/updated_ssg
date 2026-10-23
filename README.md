# Safe Sight Guardian

Safe Sight Guardian is a prototype AI-powered emergency monitoring system combining emergency hand-gesture detection with video-based action analysis.

## Features

### Phase 1: Emergency Gesture Detection

- Uses a webcam and MediaPipe hand landmarks.
- Detects the configured emergency hand-gesture sequence.
- Sends emergency events to the backend.
- Records emergency video clips for review.

### Phase 2: Video-Based Action Analysis

- Receives video clips from Phase 1.
- Uses a VD-MIL-based action-analysis pipeline.
- Displays model scores and candidate detection status.
- Saves candidate clips for human review.

**Important:** Phase 2 scores are model outputs, not calibrated probabilities of violence. A candidate detection must not be treated as a confirmed incident without human review.

### Backend

The FastAPI backend exposes endpoints for emergency events, video access, Phase 2 events, and Phase 2 status.

### Dashboard

The web dashboard displays emergency events, video recordings, and Phase 2 monitoring information.

## Project Structure

```
Safe_Sight_Guardian/
├── main.py
├── Models/
│   └── emergency_gesture_model.keras
├── hand_landmarker.task
├── backend/
│   └── server.py
├── dashboard/
│   └── index.html
├── phase 2/
│   ├── live_webcam.py
│   ├── phase2_worker.py
│   └── vdmil/
├── requirements.txt
└── .gitignore
```



## Requirements

- Python 3.11 (recommended for this project)
- Git
- A webcam
- Required model files and weights
- Python packages compatible with the Phase 1 and Phase 2 pipelines

Install the dependencies listed in `requirements.txt`:

```
python -m pip install -r requirements.txt
```

Additional backend and Phase 2 dependencies may be required; see the installation notes below.

### Install the VD-MIL MoViNet dependency

The Phase 2 action-analysis pipeline uses the MoViNet fork specified by the original VD-MIL project.
FFmpeg must also be available on `PATH` to create H.264 copies of Phase 2 clips
for browser playback. The original model clips are kept alongside those copies.

Run this command from the project root:

```
python -m pip install git+https://github.com/roggerfq/MoViNet-pytorch.git
```

Install the project's Python dependencies first:

```
python -m pip install -r requirements.txt
```

The original VD-MIL implementation is licensed under **CC BY-NC 4.0**. Review its license before redistribution or use, especially for commercial purposes.

## Running the Project

### Public read-only demo (Render)

This repository includes a Render Blueprint in `render.yaml`. To publish the
dashboard, push the project to GitHub, sign in to Render, choose **New → Blueprint**,
connect this repository, and apply the Blueprint. Render builds a lightweight
web-only environment from `requirements-deploy.txt`; the local webcam and AI
workers are not started. The service's root URL opens the dashboard.

The public demo runs in read-only mode. Local emergency footage, candidate
clips, and event files are excluded from Git, so they are not copied to the
host. The hosted dashboard therefore shows the interface without your local
recorded data or live camera. Render's free web services can spin down while
idle, and their local files do not persist across restarts. Add protected,
persistent media storage before sharing recorded footage or relying on saved
events online.

### 1. Start the backend

From the project root:

```
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Backend health endpoint:

`http://127.0.0.1:8000/health`

### 2. Open the dashboard

With the backend running, open `http://127.0.0.1:8000/dashboard/` in your browser.
The backend serves the page, event APIs, and saved emergency and candidate videos.
Avoid opening `dashboard/index.html` directly or through Live Server; that sends
API and video requests to the wrong address.

### 3. Start Phase 1

Open a separate terminal in the project root and run:

```
python main.py
```

Phase 1 is responsible for webcam processing and integration with the Phase 2 worker.

### 4. Verify operation

Check that:

- The backend health endpoint responds.
- Phase 1 detects the configured emergency gesture.
- Emergency events reach the dashboard.
- Phase 2 reports its worker status.
- Candidate clips are saved when the detection conditions are met.

If a component fails to start, inspect the terminal output and verify that its dependencies and model files are available.

## Recording and Privacy

Emergency recordings and Phase 2 candidate clips may contain sensitive footage. Keep them out of public repositories and restrict access to authorized reviewers.

The `.gitignore` file excludes configured recording directories, temporary files, and runtime event data.

## Limitations

- This is a prototype and requires testing under realistic conditions.
- Gesture and action detection can produce false positives and false negatives.
- Model scores alone do not establish that violence or an emergency occurred.
- Human review is required before acting on Phase 2 candidate detections.
- The system should not be treated as a replacement for emergency services.



## License

Add an appropriate license after confirming the licensing terms of all third-party code and models included in this project.
