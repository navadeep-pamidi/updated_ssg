# ============================================================

# Safe Sight Guardian - AI emergency hand sign detection prototype

# ============================================================



import cv2

import time

import os

import shutil

import subprocess

import math

import traceback
import glob
import json

from collections import deque



import numpy as np

import mediapipe as mp

import tensorflow as tf

import requests





# ============================================================

# PROJECT PATHS

# ============================================================



BASE_DIR = os.path.dirname(os.path.abspath(__file__))



MODEL_PATH = os.path.join(

    BASE_DIR,

    "models",

    "emergency_gesture_model.keras"

)



HAND_MODEL_PATH = os.path.join(

    BASE_DIR,

    "hand_landmarker.task"

)



VIDEO_FOLDER = os.path.join(

    BASE_DIR,

    "emergency_clips"

)



BACKEND_URL = "http://127.0.0.1:8000/emergency"

# ============================================================
# PHASE 2 - VD-MIL BRIDGE (shares Phase 1 webcam frames)
# ============================================================
PHASE2_FOLDER = os.path.join(BASE_DIR, "phase 2")
PHASE2_INCOMING_DIR = os.path.join(PHASE2_FOLDER, "incoming_clips")
PHASE2_CANDIDATE_DIR = os.path.join(PHASE2_FOLDER, "candidate_clips")
PHASE2_STATUS_FILE = os.path.join(PHASE2_CANDIDATE_DIR, "status.json")
PHASE2_WORKER_SCRIPT = os.path.join(PHASE2_FOLDER, "phase2_worker.py")

# The Phase 2 worker source is embedded in this file so you only need to maintain main.py.
PHASE2_WORKER_SOURCE = '"""Safe Sight Guardian Phase 2 VD-MIL worker.\n\nConsumes clips created by Phase 1 from its existing webcam frames. It never\nopens a webcam. Candidate clips are saved for human review, not treated as\nconfirmed violence.\n"""\nimport glob\nimport json\nimport os\nimport shutil\nimport time\nimport traceback\nfrom datetime import datetime\n\nimport numpy as np\n\nBASE_DIR = os.path.dirname(os.path.abspath(__file__))\nVDMIL_DIR = os.path.join(BASE_DIR, "vdmil")\nINCOMING_DIR = os.path.join(BASE_DIR, "incoming_clips")\nCANDIDATE_DIR = os.path.join(BASE_DIR, "candidate_clips")\nEVENTS_FILE = os.path.join(CANDIDATE_DIR, "events.jsonl")\nSTATUS_FILE = os.path.join(CANDIDATE_DIR, "status.json")\nMODEL_FOLDER = os.path.join(VDMIL_DIR, "movinet_weights")\nDETECTOR_MODEL = os.path.join(VDMIL_DIR, "test", "model_20.pt")\n\nTARGET_FPS = 8\nLENGTH_WINDOW = 8\nSTRIDE = 4\nCANDIDATE_THRESHOLD = 0.70\n# Detection uses the current clip score, NOT the smoothed display score.\n# Require two consecutive overlapping clips above threshold to reduce spikes\n# while avoiding the previous 3-window + EMA delay.\nMIN_CONSECUTIVE_HIGH_CLIPS = 2\nPOLL_SECONDS = 0.5\nDISPLAY_EMA_ALPHA = 0.30\n\nos.makedirs(INCOMING_DIR, exist_ok=True)\nos.makedirs(CANDIDATE_DIR, exist_ok=True)\n\n\ndef write_status(**values):\n    payload = {\n        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),\n        **values,\n    }\n    temp_path = STATUS_FILE + ".tmp"\n    try:\n        with open(temp_path, "w", encoding="utf-8") as f:\n            json.dump(payload, f, ensure_ascii=False)\n        os.replace(temp_path, STATUS_FILE)\n    except OSError as exc:\n        print("Could not write Phase 2 status:", exc, flush=True)\n\n\ndef as_unit_score(value):\n    """Normalize common model output scales to 0..1 for display/thresholding.\n\n    If outputs are already probabilities/scores in 0..1 they are unchanged.\n    If they are percentages in 0..100 they are converted. Other score ranges\n    are clipped, so inspect model semantics before interpreting as probability.\n    """\n    value = float(value)\n    if 1.0 < value <= 100.0:\n        value /= 100.0\n    return float(np.clip(value, 0.0, 1.0))\n\n\ndef main():\n    import sys\n    if VDMIL_DIR not in sys.path:\n        sys.path.insert(0, VDMIL_DIR)\n\n    from movinet_classifier import MovinetClassifier\n    from activity_detection import run_video_processing\n    try:\n        from signal_smoothing import smooth_scores\n    except Exception:\n        smooth_scores = None\n\n    write_status(state="STARTING", activity_score=0.0, max_score=0.0,\n                 raw_max_score=0.0, clip_score=0.0, mean_score=0.0,\n                 message="Loading VD-MIL model")\n    print("Safe Sight Guardian Phase 2 worker starting on CPU.", flush=True)\n    print("Model folder:", MODEL_FOLDER, flush=True)\n    classifier = MovinetClassifier(MODEL_FOLDER, [DETECTOR_MODEL], device="cpu")\n\n    stable_score = None\n    previous_high_clips = 0\n    write_status(state="ONLINE", activity_score=0.0, max_score=0.0,\n                 raw_max_score=0.0, clip_score=0.0, mean_score=0.0,\n                 consecutive_high_clips=0,\n                 message="Waiting for analysis window")\n    print("VD-MIL loaded. Waiting for clips; no webcam will be opened.", flush=True)\n\n    while True:\n        files = sorted(glob.glob(os.path.join(INCOMING_DIR, "analysis_*.mp4")))\n        if not files:\n            time.sleep(POLL_SECONDS)\n            continue\n\n        clip_path = files[0]\n        write_status(\n            state="PROCESSING",\n            activity_score=round(float(stable_score or 0.0), 4),\n            max_score=round(float(stable_score or 0.0), 4),\n            consecutive_high_clips=previous_high_clips,\n            message=f"Analyzing {os.path.basename(clip_path)}",\n        )\n        try:\n            scores = run_video_processing(\n                clip_path, classifier, TARGET_FPS, LENGTH_WINDOW, STRIDE\n            )\n            raw_values = np.asarray(scores, dtype=float).reshape(-1)\n            raw_values = raw_values[np.isfinite(raw_values)]\n            if raw_values.size == 0:\n                previous_high_clips = 0\n                write_status(state="MONITORING", message="Model returned no usable scores",\n                             consecutive_high_clips=0)\n                print("VD-MIL returned no usable scores for", os.path.basename(clip_path), flush=True)\n                continue\n\n            # Preserve original values for diagnostics; normalize scores for\n            # consistent display and threshold comparison.\n            normalized_raw = np.asarray([as_unit_score(v) for v in raw_values], dtype=float)\n            analysed_values = normalized_raw\n            if smooth_scores is not None:\n                try:\n                    smoothed = np.asarray(smooth_scores(normalized_raw), dtype=float).reshape(-1)\n                    smoothed = smoothed[np.isfinite(smoothed)]\n                    if smoothed.size:\n                        analysed_values = np.asarray([as_unit_score(v) for v in smoothed], dtype=float)\n                except Exception as smooth_exc:\n                    print("Optional signal smoothing failed; using raw scores:", smooth_exc, flush=True)\n                    analysed_values = normalized_raw\n\n            raw_max_score = float(np.max(normalized_raw))\n            raw_mean_score = float(np.mean(normalized_raw))\n            # A robust clip score avoids letting one isolated spike dominate.\n            clip_score = float(np.percentile(analysed_values, 75))\n            mean_score = float(np.mean(analysed_values))\n\n            # UI-only EMA. This value must NEVER gate candidate detection.\n            if stable_score is None:\n                stable_score = clip_score\n            else:\n                stable_score = (DISPLAY_EMA_ALPHA * clip_score\n                                + (1.0 - DISPLAY_EMA_ALPHA) * stable_score)\n            activity_score = float(np.clip(stable_score, 0.0, 1.0))\n\n            # Use current clip evidence for detection, independent of the EMA.\n            # If a clip\'s robust score reaches the threshold, count it as high.\n            if clip_score >= CANDIDATE_THRESHOLD:\n                previous_high_clips += 1\n            else:\n                previous_high_clips = 0\n            is_candidate = previous_high_clips >= MIN_CONSECUTIVE_HIGH_CLIPS\n\n            write_status(\n                state="CANDIDATE" if is_candidate else "MONITORING",\n                activity_score=round(activity_score, 4),\n                max_score=round(activity_score, 4),  # old UI compatibility\n                raw_max_score=round(raw_max_score, 4),\n                raw_mean_score=round(raw_mean_score, 4),\n                clip_score=round(clip_score, 4),\n                mean_score=round(mean_score, 4),\n                decision_score=round(clip_score, 4),\n                consecutive_high_clips=int(previous_high_clips),\n                threshold=CANDIDATE_THRESHOLD,\n                candidate_save_status=("PENDING_SAVE" if is_candidate else "NOT_TRIGGERED"),\n                candidate_file="",\n                candidate_bytes=0,\n                message=("Candidate threshold met; saving review clip"\n                         if is_candidate else "No candidate in latest window; no review clip saved"),\n            )\n            print(\n                f"VD-MIL raw_max={raw_max_score:.3f}, clip_p75={clip_score:.3f}, "\n                f"stable_display={activity_score:.3f}, mean={mean_score:.3f}, "\n                f"high_clip_streak={previous_high_clips}", flush=True,\n            )\n\n            if is_candidate:\n                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")\n                candidate_name = f"candidate_{stamp}.mp4"\n                candidate_path = os.path.join(CANDIDATE_DIR, candidate_name)\n                try:\n                    shutil.copy2(clip_path, candidate_path)\n                    candidate_saved = os.path.isfile(candidate_path) and os.path.getsize(candidate_path) > 0\n                    candidate_bytes = os.path.getsize(candidate_path) if candidate_saved else 0\n                    if not candidate_saved:\n                        raise IOError("Candidate clip is missing or empty after copy")\n                except Exception as save_exc:\n                    write_status(\n                        state="SAVE_ERROR",\n                        activity_score=round(activity_score, 4),\n                        decision_score=round(clip_score, 4),\n                        clip_score=round(clip_score, 4),\n                        threshold=CANDIDATE_THRESHOLD,\n                        consecutive_high_clips=int(previous_high_clips),\n                        candidate_save_status="FAILED",\n                        candidate_file=os.path.abspath(candidate_path),\n                        candidate_bytes=0,\n                        message=f"Candidate detected but video save FAILED: {save_exc}",\n                    )\n                    print(\n                        f"PHASE 2 SAVE FAILED: score={clip_score:.3f}; path={candidate_path}; error={save_exc}",\n                        flush=True,\n                    )\n                    previous_high_clips = 0\n                    continue\n\n                print(\n                    f"PHASE 2 VIDEO SAVE VERIFIED: exists={candidate_saved}, bytes={candidate_bytes}, path={os.path.abspath(candidate_path)}",\n                    flush=True,\n                )\n                write_status(\n                    state="CANDIDATE",\n                    activity_score=round(activity_score, 4),\n                    decision_score=round(clip_score, 4),\n                    clip_score=round(clip_score, 4),\n                    threshold=CANDIDATE_THRESHOLD,\n                    consecutive_high_clips=int(previous_high_clips),\n                    candidate_save_status="SAVED",\n                    candidate_file=os.path.abspath(candidate_path),\n                    candidate_bytes=int(candidate_bytes),\n                    candidate_saved_at=datetime.now().astimezone().isoformat(timespec="seconds"),\n                    message="Candidate detected; video saved and verified for human review",\n                )\n                event = {\n                    "event_type": "PHASE2_CANDIDATE",\n                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),\n                    "activity_score": round(activity_score, 4),\n                    "decision_score": round(clip_score, 4),\n                    "clip_score": round(clip_score, 4),\n                    "raw_max_score": round(raw_max_score, 4),\n                    "raw_mean_score": round(raw_mean_score, 4),\n                    "mean_score": round(mean_score, 4),\n                    "consecutive_high_clips": int(previous_high_clips),\n                    "threshold": CANDIDATE_THRESHOLD,\n                    "clip_path": os.path.abspath(candidate_path),\n                    "clip_filename": candidate_name,\n                    "review_status": "pending_human_review",\n                    "note": "Automated model candidate only; human review required. Score is not a calibrated probability of violence.",\n                }\n                with open(EVENTS_FILE, "a", encoding="utf-8") as f:\n                    f.write(json.dumps(event, ensure_ascii=False) + "\\n")\n                print("PHASE 2 CANDIDATE SAVED FOR HUMAN REVIEW:", candidate_path, flush=True)\n                # Reset streak after saving so a sustained event does not save\n                # every subsequent clip as a new candidate.\n                previous_high_clips = 0\n        except Exception as exc:\n            write_status(state="ERROR", message=str(exc))\n            print("Phase 2 processing error:", exc, flush=True)\n            traceback.print_exc()\n        finally:\n            try:\n                if os.path.exists(clip_path):\n                    os.remove(clip_path)\n            except OSError:\n                pass\n\n\nif __name__ == "__main__":\n    try:\n        main()\n    except KeyboardInterrupt:\n        print("Phase 2 worker stopped.", flush=True)\n    except Exception as exc:\n        try:\n            write_status(state="ERROR", message=str(exc))\n        except Exception:\n            pass\n        print("Phase 2 worker could not start:", exc, flush=True)\n        traceback.print_exc()\n'
# The VD-MIL dependencies are kept in their separate virtual environment.
# Legacy setting: set NEXA_PHASE2_PYTHON if the Phase 2 environment lives elsewhere.
# Fall back to the current interpreter so the worker can report a visible
# dependency error in the OpenCV panel instead of silently disabling Phase 2.
_phase2_python_candidates = [
    os.environ.get("NEXA_PHASE2_PYTHON", ""),
    os.path.join(os.path.expanduser("~"), "venvs", "phase2-vdmil", "Scripts", "python.exe"),
    os.path.join(PHASE2_FOLDER, ".venv", "Scripts", "python.exe"),
    os.path.join(PHASE2_FOLDER, "venv", "Scripts", "python.exe"),
    os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe"),
    os.path.join(BASE_DIR, "venv", "Scripts", "python.exe"),
]
PHASE2_PYTHON = next(
    (candidate for candidate in _phase2_python_candidates if candidate and os.path.isfile(candidate)),
    __import__("sys").executable,
)
PHASE2_PYTHON_IS_FALLBACK = not any(
    candidate and os.path.isfile(candidate) for candidate in _phase2_python_candidates
)
PHASE2_FPS = 8
PHASE2_WINDOW_SECONDS = 8
PHASE2_CLIP_INTERVAL_SECONDS = 4
PHASE2_BUFFER_FRAMES = PHASE2_FPS * PHASE2_WINDOW_SECONDS

os.makedirs(PHASE2_INCOMING_DIR, exist_ok=True)
os.makedirs(PHASE2_CANDIDATE_DIR, exist_ok=True)

# Create the worker script from the embedded source only for older checkouts
# that do not yet contain the maintained standalone worker.
try:
    if not os.path.isfile(PHASE2_WORKER_SCRIPT):
        with open(PHASE2_WORKER_SCRIPT, "w", encoding="utf-8") as phase2_worker_file:
            phase2_worker_file.write(PHASE2_WORKER_SOURCE)
except OSError as phase2_file_error:
    print("Could not prepare Phase 2 worker; Phase 1 will continue:", phase2_file_error)






# ============================================================

# CONFIGURATION

# ============================================================



# Existing ML model threshold for SIGN 2

EMERGENCY_THRESHOLD = 0.90



# Sign 2 must remain valid for this long

REQUIRED_TIME = 2.0



# Sign 2 must occur within this time after Sign 1

SEQUENCE_TIMEOUT = 5.0



# How long EMERGENCY should remain on screen

EMERGENCY_DISPLAY_TIME = 15.0  # Keep Phase 1 emergency status visible for 15 seconds



# Video before emergency

BUFFER_SECONDS = 5



# Video after emergency

RECORD_AFTER_SECONDS = 15



# Minimum time between recordings

RECORD_COOLDOWN = 10





# ============================================================

# SIGN 1 CONFIGURATION

#

# SIGN 1:

#

# Index   = OPEN

# Middle  = OPEN

# Ring    = OPEN

# Pinky   = OPEN

# Thumb   = FOLDED

# ============================================================



SIGN1_TIMEOUT = SEQUENCE_TIMEOUT



# Sign 1 must be seen on this many consecutive frames before

# the Sign 1 -> Sign 2 sequence starts. This removes one-frame

# false positives caused by landmark jitter.

SIGN1_CONFIRM_FRAMES = 3



# ---- Finger "open" thresholds (degrees / ratios) ----

# Angle at the PIP joint (MCP-PIP-DIP). \~180 = straight finger,

# \~60-100 = curled finger.

FINGER_PIP_ANGLE_MIN = 160.0



# Angle at the DIP joint (PIP-DIP-TIP).

FINGER_DIP_ANGLE_MIN = 150.0



# Fingertip must be at least this many times farther from the

# wrist than the finger's own MCP knuckle.

FINGER_TIP_TO_MCP_RATIO_MIN = 1.35



# ---- Thumb "folded" thresholds (ratios of hand size) ----

# hand size = distance wrist (0) -> middle-finger MCP (9)

#

# Thumb tip must be close to the palm centre ...

THUMB_PALM_RATIO_MAX = 0.60



# ... and close to the base of the index finger.

THUMB_INDEX_MCP_RATIO_MAX = 0.80





# ============================================================

# CREATE VIDEO FOLDER

# ============================================================



os.makedirs(

    VIDEO_FOLDER,

    exist_ok=True

)





# ============================================================

# CHECK FFMPEG

# ============================================================



FFMPEG_PATH = shutil.which("ffmpeg")



if FFMPEG_PATH is None:



    print()

    print("❌ FFmpeg was not found.")

    print("Please install FFmpeg and restart Cursor.")

    exit()



print()

print("FFmpeg found:")

print(FFMPEG_PATH)





# ============================================================

# COLORS - BGR

# ============================================================



BG = (15, 18, 25)



PANEL = (25, 30, 40)



WHITE = (235, 240, 245)



GRAY = (130, 140, 150)



CYAN = (255, 210, 40)



GREEN = (80, 220, 120)



ORANGE = (40, 170, 255)



RED = (70, 70, 255)





# ============================================================

# LOAD AI MODEL

# ============================================================



print()

print("Loading AI model...")



model = tf.keras.models.load_model(

    MODEL_PATH

)



print(

    "AI model loaded successfully!"

)





# ============================================================

# MEDIAPIPE

# ============================================================



BaseOptions = mp.tasks.BaseOptions



HandLandmarker = (

    mp.tasks.vision.HandLandmarker

)



HandLandmarkerOptions = (

    mp.tasks.vision.HandLandmarkerOptions

)



RunningMode = (

    mp.tasks.vision.RunningMode

)





options = HandLandmarkerOptions(



    base_options=BaseOptions(

        model_asset_path=HAND_MODEL_PATH

    ),



    running_mode=RunningMode.VIDEO,



    num_hands=1,



    min_hand_detection_confidence=0.5,



    min_hand_presence_confidence=0.5,



    min_tracking_confidence=0.5

)





hand_landmarker = (

    HandLandmarker.create_from_options(

        options

    )

)





# ============================================================

# NORMALIZE LANDMARKS

# ============================================================



def normalize_landmarks(landmarks):



    landmarks = np.array(

        landmarks,

        dtype=np.float32

    ).reshape(

        21,

        3

    )



    # Wrist becomes origin



    landmarks = (

        landmarks

        - landmarks[0]

    )



    # Scale



    scale = np.max(

        np.linalg.norm(

            landmarks,

            axis=1

        )

    )



    if scale == 0:



        return None



    landmarks = (

        landmarks / scale

    )



    return landmarks.flatten()





# ============================================================

# LANDMARK DISTANCE

#

# Used by the (unchanged) Sign 2 shape check.

# ============================================================



def landmark_distance(a, b):



    return math.sqrt(



        (a.x - b.x) ** 2 +



        (a.y - b.y) ** 2 +



        (a.z - b.z) ** 2



    )





# ============================================================

# SIGN 1 GEOMETRY HELPERS

#

# WHY THE OLD DETECTOR WAS UNRELIABLE:

#

# 1. MediaPipe returns x and y normalized separately by image

#    width and height. With a 900x600 frame this stretches the

#    x axis relative to y, so "distances" were distorted and

#    changed with hand rotation.

#

# 2. Fingers were judged only by "tip is farther from the wrist

#    than the PIP joint", which is also true for many curled or

#    half-curled fingers.

#

# 3. The thumb was judged with one loose distance test.

#

# NEW APPROACH:

#

# - All geometry is computed on landmarks converted to a common

#   pixel scale (x  width, y  height, z  width), so distances*

#   and angles are not distorted by the frame's aspect ratio.

#

# - Fingers use joint ANGLES (PIP and DIP) plus a tip-to-knuckle

#   reach check. Joint angles and distance ratios do not depend

#   on which way the hand points or on mirroring of the image,

#   because they use no fixed x / y direction comparisons.

#

# - The thumb is judged by how close its tip is to the palm

#   centre and to the index-finger base, relative to hand size.

#

# LIMITATIONS (not tested against every hand / orientation):

#

# - MediaPipe z is an estimate and is noisy; angles shrink when

#   a finger points toward or away from the camera

#   (foreshortening), so an open finger aimed at the camera can

#   be rejected.

#

# - Thresholds are tuned by reasoning about typical hand

#   proportions, not by testing on a dataset. Different hand

#   sizes, camera angles or lighting may need the constants at

#   the top of the file adjusted.

#

# - A thumb pressed flat against the side of the hand can be

#   classified as folded, and a half-folded thumb near the

#   threshold may flicker between folded and not folded.

#

# - Poor hand tracking (blur, partial hand in frame) produces

#   bad landmarks, which produces bad results.

# ============================================================



def landmarks_to_points(

    hand,

    frame_width,

    frame_height

):



    return np.array(



        [

            [

                landmark.x * frame_width,

                landmark.y * frame_height,

                landmark.z * frame_width

            ]

            for landmark in hand

        ],



        dtype=np.float32



    )





def joint_angle(a, b, c):



    # Angle at point b, in degrees, between segments b->a and b->c



    ba = a - b



    bc = c - b



    norm_ba = np.linalg.norm(ba)



    norm_bc = np.linalg.norm(bc)



    if norm_ba == 0 or norm_bc == 0:



        return 0.0



    cosine = float(



        np.dot(ba, bc)

        / (norm_ba * norm_bc)



    )



    cosine = max(

        -1.0,

        min(1.0, cosine)

    )



    return math.degrees(

        math.acos(cosine)

    )





# ============================================================

# CHECK WHETHER FINGER IS EXTENDED

#

# A finger is open only if ALL of these are true:

#

# 1. PIP joint is nearly straight

# 2. DIP joint is nearly straight

# 3. Fingertip reaches well beyond the finger's knuckle (MCP),

#    measured from the wrist

# 4. Fingertip is farther from the wrist than the PIP joint

# ============================================================



def finger_is_extended(

    points,

    mcp_index,

    pip_index,

    dip_index,

    tip_index

):



    wrist = points[0]



    mcp = points[mcp_index]



    pip = points[pip_index]



    dip = points[dip_index]



    tip = points[tip_index]





    pip_angle = joint_angle(

        mcp,

        pip,

        dip

    )



    dip_angle = joint_angle(

        pip,

        dip,

        tip

    )





    tip_to_wrist = float(

        np.linalg.norm(tip - wrist)

    )



    pip_to_wrist = float(

        np.linalg.norm(pip - wrist)

    )



    mcp_to_wrist = float(

        np.linalg.norm(mcp - wrist)

    )





    if mcp_to_wrist == 0:



        return False





    return bool(



        pip_angle >= FINGER_PIP_ANGLE_MIN



        and dip_angle >= FINGER_DIP_ANGLE_MIN



        and tip_to_wrist > pip_to_wrist



        and (

            tip_to_wrist / mcp_to_wrist

            >= FINGER_TIP_TO_MCP_RATIO_MIN

        )



    )





# ============================================================

# CHECK WHETHER THUMB IS FOLDED

#

# The thumb counts as folded only when its tip is close to the

# palm centre AND close to the base of the index finger.

# An extended thumb points away from the palm, so its tip is

# far from both points regardless of hand rotation.

# ============================================================



def thumb_is_folded(points):



    hand_size = float(



        np.linalg.norm(

            points[9] - points[0]

        )



    )



    if hand_size == 0:



        return False





    palm_center = np.mean(



        points[[0, 5, 9, 13, 17]],



        axis=0



    )



    thumb_tip = points[4]





    tip_to_palm = float(



        np.linalg.norm(

            thumb_tip - palm_center

        )



    ) / hand_size





    tip_to_index_mcp = float(



        np.linalg.norm(

            thumb_tip - points[5]

        )



    ) / hand_size





    return bool(



        tip_to_palm <= THUMB_PALM_RATIO_MAX



        and



        tip_to_index_mcp

        <= THUMB_INDEX_MCP_RATIO_MAX



    )





# ============================================================

# SIGN 1 DETECTOR

#

# Index   = OPEN

# Middle  = OPEN

# Ring    = OPEN

# Pinky   = OPEN

# Thumb   = FOLDED

#

# Strict result: every condition must be true at once.

# ============================================================



def detect_sign1(

    hand,

    frame_width=900,

    frame_height=600

):



    if hand is None or len(hand) < 21:



        return False





    points = landmarks_to_points(

        hand,

        frame_width,

        frame_height

    )





    # Index finger  : landmarks 5, 6, 7, 8

    index_open = finger_is_extended(

        points,

        5,

        6,

        7,

        8

    )



    # Middle finger : landmarks 9, 10, 11, 12

    middle_open = finger_is_extended(

        points,

        9,

        10,

        11,

        12

    )



    # Ring finger   : landmarks 13, 14, 15, 16

    ring_open = finger_is_extended(

        points,

        13,

        14,

        15,

        16

    )



    # Pinky         : landmarks 17, 18, 19, 20

    pinky_open = finger_is_extended(

        points,

        17,

        18,

        19,

        20

    )



    # Thumb         : landmarks 1, 2, 3, 4

    thumb_folded = thumb_is_folded(

        points

    )





    sign1_detected = (



        index_open



        and middle_open



        and ring_open



        and pinky_open



        and thumb_folded



    )





    return bool(sign1_detected)





# ============================================================

# SIGN 2 DETECTOR

#

# ALL FOUR FINGERS MUST BE CLOSED

# AND THUMB MUST BE CLOSED.

#

# This prevents Sign 1 from being accepted as Sign 2

# even when the ML model gives high confidence.

# ============================================================



def all_fingers_closed(hand):



    # --------------------------------------------------------

    # INDEX

    # --------------------------------------------------------



    index_closed = (

        hand[8].y > hand[6].y

    )





    # --------------------------------------------------------

    # MIDDLE

    # --------------------------------------------------------



    middle_closed = (

        hand[12].y > hand[10].y

    )





    # --------------------------------------------------------

    # RING

    # --------------------------------------------------------



    ring_closed = (

        hand[16].y > hand[14].y

    )





    # --------------------------------------------------------

    # PINKY

    # --------------------------------------------------------



    pinky_closed = (

        hand[20].y > hand[18].y

    )





    # --------------------------------------------------------

    # THUMB

    # --------------------------------------------------------



    thumb_tip = hand[4]



    thumb_mcp = hand[2]



    thumb_closed = (



        landmark_distance(

            thumb_tip,

            thumb_mcp

        )



        <



        0.12



    )





    # --------------------------------------------------------

    # ALL FINGERS MUST BE CLOSED

    # --------------------------------------------------------



    return (



        index_closed



        and middle_closed



        and ring_closed



        and pinky_closed



        and thumb_closed



    )





# ============================================================

# DRAW TEXT

# ============================================================



def text(

    frame,

    message,

    position,

    size=0.6,

    color=WHITE,

    thickness=1

):



    cv2.putText(



        frame,



        message,



        position,



        cv2.FONT_HERSHEY_SIMPLEX,



        size,



        color,



        thickness,



        cv2.LINE_AA



    )





# ============================================================

# DRAW PANEL

# ============================================================



def panel(

    frame,

    x1,

    y1,

    x2,

    y2

):



    cv2.rectangle(



        frame,



        (x1, y1),



        (x2, y2),



        PANEL,



        -1



    )



    cv2.rectangle(



        frame,



        (x1, y1),



        (x2, y2),



        (55, 65, 80),



        1



    )





# ============================================================

# SEND EMERGENCY ALERT

# ============================================================



def send_emergency_alert(

    video_filename

):



    try:



        data = {

            "video": video_filename

        }



        response = requests.post(



            BACKEND_URL,



            json=data,



            timeout=5



        )



        if response.status_code == 200:



            print()



            print(

                "🚨 Emergency alert sent to backend!"

            )



            return True





        print(

            "❌ Backend returned:",

            response.status_code

        )



        return False





    except requests.exceptions.RequestException as e:



        print(

            "❌ Could not connect to backend:",

            e

        )



        return False





# ============================================================

# CONVERT VIDEO TO H.264

# ============================================================



def convert_to_h264(

    raw_filepath,

    final_filepath

):



    print()



    print("=" * 60)



    print("🎬 Converting video to H.264...")



    print("=" * 60)





    command = [



        FFMPEG_PATH,



        "-y",



        "-i",

        raw_filepath,



        "-c:v",

        "libx264",



        "-preset",

        "fast",



        "-crf",

        "23",



        "-pix_fmt",

        "yuv420p",



        "-movflags",

        "+faststart",



        final_filepath



    ]





    try:



        result = subprocess.run(



            command,



            stdout=subprocess.PIPE,



            stderr=subprocess.PIPE,



            text=True



        )





        if result.returncode != 0:



            print(

                "❌ FFmpeg conversion failed."

            )



            print(

                result.stderr

            )



            return False





        print(

            "✅ H.264 video created."

        )





        if os.path.exists(

            raw_filepath

        ):



            os.remove(

                raw_filepath

            )





        return True





    except Exception as e:



        print(

            "❌ FFmpeg error:",

            e

        )



        return False





# ============================================================

# FINALIZE RECORDING

# ============================================================



def finalize_recording(

    video_writer,

    raw_filepath,

    final_filepath

):



    # --------------------------------------------------------

    # RELEASE WRITER

    # --------------------------------------------------------



    if video_writer is not None:



        video_writer.release()



        print(

            "Video writer closed."

        )





    # --------------------------------------------------------

    # CHECK RAW FILE

    # --------------------------------------------------------



    if not os.path.exists(

        raw_filepath

    ):



        print(

            "❌ Raw video file does not exist."

        )



        return False





    raw_size = os.path.getsize(

        raw_filepath

    )





    print(

        f"Raw video size: "

        f"{raw_size / 1024 / 1024:.2f} MB"

    )





    if raw_size == 0:



        print(

            "❌ Raw video is empty."

        )



        return False





    # --------------------------------------------------------

    # CONVERT

    # --------------------------------------------------------



    success = convert_to_h264(



        raw_filepath,



        final_filepath



    )





    if not success:



        return False





    # --------------------------------------------------------

    # VERIFY FINAL FILE

    # --------------------------------------------------------



    if not os.path.exists(

        final_filepath

    ):



        print(

            "❌ Final video was not created."

        )



        return False





    final_size = os.path.getsize(

        final_filepath

    )





    print(

        f"Final video size: "

        f"{final_size / 1024 / 1024:.2f} MB"

    )





    if final_size == 0:



        print(

            "❌ Final video is empty."

        )



        return False





    print()



    print(

        "🎉 Emergency video completely saved!"

    )



    print(

        "📁",

        final_filepath

    )



    return True





# ============================================================

# CAMERA

# ============================================================



cap = cv2.VideoCapture(0)





if not cap.isOpened():



    print(

        "❌ ERROR: Could not open webcam."

    )



    hand_landmarker.close()



    exit()





cap.set(

    cv2.CAP_PROP_FRAME_WIDTH,

    1280

)



cap.set(

    cv2.CAP_PROP_FRAME_HEIGHT,

    720

)





print()



print(

    "Camera started."

)



print(

    "Press Q or ESC to quit."

)





# ============================================================

# FPS

# ============================================================



fps = cap.get(

    cv2.CAP_PROP_FPS

)





if fps <= 0 or fps > 120:



    fps = 30





print(

    "Camera FPS:",

    fps

)





# ============================================================

# VIDEO BUFFER

# ============================================================



max_buffer_frames = int(

    BUFFER_SECONDS * fps

)



frame_buffer = deque(

    maxlen=max_buffer_frames

)

# Phase 2 receives sampled copies of the SAME frames read by Phase 1.
phase2_frame_buffer = deque(maxlen=PHASE2_BUFFER_FRAMES)
phase2_last_sample_time = 0.0
phase2_last_clip_time = 0.0
phase2_worker_process = None
phase2_enabled = False
phase2_status_cache = {"state": "STARTING", "max_score": 0.0, "mean_score": 0.0, "candidate_save_status": "NOT_TRIGGERED", "message": "Starting worker"}
phase2_status_last_read = 0.0

# Start the worker with its own Python environment. The worker processes
# video files only; it never opens a webcam. Phase 1 still owns camera 0.
if os.path.isfile(PHASE2_PYTHON) and os.path.isfile(PHASE2_WORKER_SCRIPT):
    try:
        if PHASE2_PYTHON_IS_FALLBACK:
            print("WARNING: dedicated Phase 2 Python environment was not found.")
            print("Using current Python; if VD-MIL dependencies are missing, Phase 2 will show ERROR.")
        print("Phase 2 Python:", PHASE2_PYTHON)
        phase2_worker_process = subprocess.Popen(
            [PHASE2_PYTHON, "-u", PHASE2_WORKER_SCRIPT],
            cwd=PHASE2_FOLDER,
            stdout=None,
            stderr=None,
        )
        phase2_enabled = True
        print("Phase 2 VD-MIL worker launched (shared webcam frames; no second camera).")
    except Exception as phase2_start_error:
        phase2_status_cache.update({"state": "ERROR", "message": str(phase2_start_error)})
        print("Phase 2 could not start; Phase 1 will continue:", phase2_start_error)
else:
    missing = PHASE2_WORKER_SCRIPT if not os.path.isfile(PHASE2_WORKER_SCRIPT) else PHASE2_PYTHON
    phase2_status_cache.update({"state": "OFFLINE", "message": f"Missing: {missing}"})
    print("Phase 2 worker or Python executable missing:", missing)
    print("Phase 1 remains enabled; fix the path shown above to enable Phase 2.")





# ============================================================

# RECORDING VARIABLES

# ============================================================



recording = False



video_writer = None



recording_end_time = None



last_saved_time = 0



current_video_filename = None



current_raw_filepath = None



current_final_filepath = None





# ============================================================

# TWO-SIGN SEQUENCE VARIABLES

# ============================================================



SEQUENCE_WAITING = 0



SEQUENCE_SIGN1_DETECTED = 1





sequence_state = SEQUENCE_WAITING



sequence_start_time = None



# Consecutive frames in which Sign 1 was seen

sign1_frames = 0





# ============================================================

# EMERGENCY VARIABLES

# ============================================================



emergency_start_time = None



emergency_verified = False



# Stores when emergency was verified

emergency_verified_time = None





# ============================================================

# START TIME

# ============================================================



start_time = time.time()





# ============================================================

# OPENCV WINDOW

# ============================================================



WINDOW_NAME = "Safe Sight Guardian - Emergency AI"





cv2.namedWindow(

    WINDOW_NAME,

    cv2.WINDOW_NORMAL

)



cv2.resizeWindow(

    WINDOW_NAME,

    1500,

    760

)



cv2.moveWindow(

    WINDOW_NAME,

    50,

    50

)





# ============================================================

# MAIN LOOP

# ============================================================



try:



    while True:



        # ====================================================

        # READ CAMERA

        # ====================================================



        ret, camera_frame = cap.read()





        if not ret:



            print(

                "❌ Failed to read camera."

            )



            break





        # ====================================================

        # MIRROR

        # ====================================================



        camera_frame = cv2.flip(

            camera_frame,

            1

        )





        # ====================================================

        # RESIZE

        # ====================================================



        camera_frame = cv2.resize(



            camera_frame,



            (900, 600)



        )





        # ====================================================

        # MEDIAPIPE

        # ====================================================



        rgb_frame = cv2.cvtColor(



            camera_frame,



            cv2.COLOR_BGR2RGB



        )





        mp_image = mp.Image(



            image_format=mp.ImageFormat.SRGB,



            data=rgb_frame



        )





        timestamp_ms = int(



            (

                time.time()

                - start_time

            )

            * 1000



        )





        result = (



            hand_landmarker



            .detect_for_video(



                mp_image,



                timestamp_ms



            )



        )





        # ====================================================

        # DEFAULT VALUES

        # ====================================================



        probability = 0.0



        hand_detected = False



        sign1_detected = False



        sign2_shape_detected = False





        # ====================================================

        # HAND DETECTION

        # ====================================================



        if result.hand_landmarks:



            hand_detected = True



            hand = (

                result.hand_landmarks[0]

            )





            # =================================================

            # SIGN 1

            #

            # Geometry is evaluated in pixel scale, so the

            # actual frame size is passed in.

            # =================================================



            sign1_detected = detect_sign1(



                hand,



                camera_frame.shape[1],



                camera_frame.shape[0]



            )





            # =================================================

            # SIGN 2 SHAPE

            # =================================================



            sign2_shape_detected = (

                all_fingers_closed(hand)

            )





            # =================================================

            # CREATE LANDMARK ARRAY

            # =================================================



            landmarks = []





            for landmark in hand:



                landmarks.append(



                    [

                        landmark.x,

                        landmark.y,

                        landmark.z

                    ]



                )





            # =================================================

            # NORMALIZE

            # =================================================



            features = normalize_landmarks(

                landmarks

            )





            # =================================================

            # MODEL PREDICTION

            # =================================================



            if features is not None:



                features = np.array(

                    features

                ).reshape(

                    1,

                    63

                )





                probability = float(



                    model.predict(



                        features,



                        verbose=0



                    )[0][0]



                )





            # =================================================

            # DRAW HAND LANDMARKS

            # =================================================



            h, w, _ = (

                camera_frame.shape

            )





            points = []





            for landmark in hand:



                x = int(

                    landmark.x * w

                )



                y = int(

                    landmark.y * h

                )



                points.append(

                    (x, y)

                )





            connections = [



                (0, 1),

                (1, 2),

                (2, 3),

                (3, 4),



                (0, 5),

                (5, 6),

                (6, 7),

                (7, 8),



                (0, 9),

                (9, 10),

                (10, 11),

                (11, 12),



                (0, 13),

                (13, 14),

                (14, 15),

                (15, 16),



                (0, 17),

                (17, 18),

                (18, 19),

                (19, 20),



                (5, 9),

                (9, 13),

                (13, 17)



            ]





            for start, end in connections:



                cv2.line(



                    camera_frame,



                    points[start],



                    points[end],



                    CYAN,



                    2,



                    cv2.LINE_AA



                )





            for point in points:



                cv2.circle(



                    camera_frame,



                    point,



                    5,



                    WHITE,



                    -1,



                    cv2.LINE_AA



                )





        else:



            hand_detected = False



            sign1_detected = False



            sign2_shape_detected = False





        # ====================================================

        # CONSECUTIVE SIGN 1 FRAME COUNTER

        # ====================================================



        if sign1_detected:



            sign1_frames += 1



        else:



            sign1_frames = 0





        # ====================================================

        # ADD FRAME TO BUFFER

        # ====================================================



        frame_buffer.append(

            camera_frame.copy()

        )

        # ----------------------------------------------------
        # PHASE 2: SAMPLE THE SAME CAMERA FRAME AT 8 FPS
        # ----------------------------------------------------
        if phase2_enabled:
            phase2_now = time.monotonic()
            if phase2_now - phase2_last_sample_time >= (1.0 / PHASE2_FPS):
                phase2_frame_buffer.append(camera_frame.copy())
                phase2_last_sample_time = phase2_now

            # Keep at most one unprocessed clip in the queue. This prevents
            # CPU inference from building an ever-growing backlog.
            pending_phase2_clips = glob.glob(
                os.path.join(PHASE2_INCOMING_DIR, "analysis_*.mp4")
            )
            if (
                len(phase2_frame_buffer) == PHASE2_BUFFER_FRAMES
                and phase2_now - phase2_last_clip_time >= PHASE2_CLIP_INTERVAL_SECONDS
                and not pending_phase2_clips
            ):
                clip_stamp = time.strftime("%Y%m%d_%H%M%S")
                writing_path = os.path.join(
                    PHASE2_INCOMING_DIR, f"_writing_{clip_stamp}.mp4"
                )
                ready_path = os.path.join(
                    PHASE2_INCOMING_DIR, f"analysis_{clip_stamp}.mp4"
                )
                phase2_writer = cv2.VideoWriter(
                    writing_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    PHASE2_FPS,
                    (camera_frame.shape[1], camera_frame.shape[0])
                )
                if phase2_writer.isOpened():
                    for phase2_frame in list(phase2_frame_buffer):
                        phase2_writer.write(phase2_frame)
                    phase2_writer.release()
                    if os.path.exists(writing_path) and os.path.getsize(writing_path) > 0:
                        os.replace(writing_path, ready_path)
                        phase2_last_clip_time = phase2_now
                        print("Phase 2 clip queued:", os.path.basename(ready_path))
                    else:
                        print("Phase 2 clip was empty; skipping this analysis window.")
                else:
                    phase2_writer.release()
                    print("Phase 2 clip writer failed; Phase 1 continues normally.")
                    try:
                        if os.path.exists(writing_path):
                            os.remove(writing_path)
                    except OSError:
                        pass





        # ====================================================

        # CURRENT TIME

        # ====================================================



        current_time = time.time()

        # Detect worker crashes instead of leaving a misleading ON status.
        if phase2_worker_process is not None and phase2_worker_process.poll() is not None:
            phase2_enabled = False
            phase2_status_cache.update({
                "state": "ERROR",
                "message": f"Worker exited (code {phase2_worker_process.returncode})",
            })

        # Refresh Phase 2 worker status without blocking the camera loop.
        if current_time - phase2_status_last_read >= 0.5:
            phase2_status_last_read = current_time
            try:
                if os.path.isfile(PHASE2_STATUS_FILE):
                    with open(PHASE2_STATUS_FILE, "r", encoding="utf-8") as status_file:
                        loaded_phase2_status = json.load(status_file)
                    if isinstance(loaded_phase2_status, dict) and not (
                        phase2_worker_process is not None and phase2_worker_process.poll() is not None
                    ):
                        phase2_status_cache.update(loaded_phase2_status)
                elif not phase2_enabled:
                    phase2_status_cache.update({"state": "OFFLINE", "message": "Worker not started"})
            except (OSError, json.JSONDecodeError):
                pass





        # ====================================================

        # RESET EMERGENCY DISPLAY STATUS

        #

        # After 5 seconds, return to normal operation.

        #

        # This only resets the on-screen emergency state and

        # the detection sequence. It does NOT touch the

        # recording variables, the video writer, the FFmpeg

        # conversion, or the backend alert, so those finish

        # normally in the RECORDING section below.

        # ====================================================



        if (



            emergency_verified



            and



            emergency_verified_time is not None



            and



            (

                current_time

                - emergency_verified_time

            )



            >= EMERGENCY_DISPLAY_TIME



        ):



            emergency_verified = False



            emergency_verified_time = None



            sequence_state = (

                SEQUENCE_WAITING

            )



            sequence_start_time = None



            emergency_start_time = None



            sign1_frames = 0





            print()



            print(

                "✅ Emergency status cleared."

            )



            print(

                "System returned to normal monitoring."

            )





        # ====================================================

        # TWO-SIGN SEQUENCE

        # ====================================================



        if (



            sequence_state

            == SEQUENCE_WAITING



            and



            not emergency_verified



        ):



            emergency_start_time = None





            # ------------------------------------------------

            # SIGN 1 must be stable for several frames

            # ------------------------------------------------



            if (



                sign1_detected



                and



                sign1_frames

                >= SIGN1_CONFIRM_FRAMES



            ):



                sequence_state = (

                    SEQUENCE_SIGN1_DETECTED

                )





                sequence_start_time = (

                    current_time

                )





                print()



                print("=" * 60)



                print(

                    "✋ SIGN 1 DETECTED"

                )



                print(

                    "Waiting for emergency SIGN 2..."

                )



                print("=" * 60)





        # ====================================================

        # STATE 1

        # SIGN 1 WAS DETECTED

        # ====================================================



        elif (



            sequence_state

            == SEQUENCE_SIGN1_DETECTED



            and



            not emergency_verified



        ):



            sequence_elapsed = (



                current_time

                - sequence_start_time



            )





            # ------------------------------------------------

            # SIGN 2 VALIDITY

            #

            # BOTH CONDITIONS MUST BE TRUE:

            #

            # 1. ML confidence >= 90%

            # 2. ALL fingers are physically closed

            #

            # High ML probability alone is NOT enough.

            # ------------------------------------------------



            sign2_valid = (



                probability

                >= EMERGENCY_THRESHOLD



                and



                sign2_shape_detected



            )





            # ------------------------------------------------

            # TIMEOUT

            #

            # The 5 second window is for WAITING for Sign 2.

            # A verification already in progress is not cut

            # off by the timeout; it ends only when it

            # completes or when Sign 2 becomes invalid.

            # ------------------------------------------------



            if (



                emergency_start_time is None



                and



                sequence_elapsed

                > SEQUENCE_TIMEOUT



            ):



                print()



                print(

                    "⏱️ Sequence timeout."

                )



                print(

                    "SIGN 2 was not detected."

                )





                sequence_state = (

                    SEQUENCE_WAITING

                )



                sequence_start_time = None



                emergency_start_time = None



                sign1_frames = 0





            # ------------------------------------------------

            # SIGN 2 VALID

            # ------------------------------------------------



            elif sign2_valid:



                if emergency_start_time is None:



                    emergency_start_time = (

                        current_time

                    )





                    print()



                    print(

                        "🚨 SIGN 2 DETECTED!"

                    )



                    print(

                        "Starting final verification..."

                    )





                elapsed = (



                    current_time

                    - emergency_start_time



                )





                # ------------------------------------------------

                # 2 SECOND VERIFICATION COMPLETE

                # ------------------------------------------------



                if elapsed >= REQUIRED_TIME:



                    emergency_verified = True



                    emergency_verified_time = (

                        current_time

                    )





                    print()



                    print("=" * 60)



                    print(

                        "🚨 EMERGENCY VERIFIED!"

                    )



                    print(

                        "SIGN 1 → SIGN 2 sequence completed."

                    )



                    print("=" * 60)





                    # ====================================================

                    # START RECORDING

                    # ====================================================



                    if (



                        not recording



                        and



                        (

                            current_time

                            - last_saved_time

                        )

                        >= RECORD_COOLDOWN



                    ):



                        timestamp = time.strftime(

                            "%Y%m%d_%H%M%S"

                        )





                        current_video_filename = (



                            f"emergency_"

                            f"{timestamp}.mp4"



                        )





                        current_final_filepath = os.path.join(



                            VIDEO_FOLDER,



                            current_video_filename



                        )





                        current_raw_filepath = os.path.join(



                            VIDEO_FOLDER,



                            f"temp_{timestamp}.mp4"



                        )





                        # ------------------------------------------------

                        # VIDEO WRITER

                        # ------------------------------------------------



                        fourcc = cv2.VideoWriter_fourcc(

                            *"mp4v"

                        )





                        video_writer = cv2.VideoWriter(



                            current_raw_filepath,



                            fourcc,



                            fps,



                            (

                                camera_frame.shape[1],

                                camera_frame.shape[0]

                            )



                        )





                        # ------------------------------------------------

                        # WRITE BUFFER

                        # ------------------------------------------------



                        for buffered_frame in frame_buffer:



                            video_writer.write(

                                buffered_frame

                            )





                        recording = True





                        recording_end_time = (



                            current_time

                            + RECORD_AFTER_SECONDS



                        )





                        last_saved_time = (

                            current_time

                        )





                        print()



                        print(

                            "🎥 Recording emergency video..."

                        )



                        print(

                            "File:",

                            current_final_filepath

                        )





            # ------------------------------------------------

            # SIGN 2 NOT VALID

            #

            # Verification timer restarts from zero.

            # ------------------------------------------------



            else:



                emergency_start_time = None





        # ====================================================

        # RECORDING

        #

        # Independent of the emergency display state. Clearing

        # the EMERGENCY indicator never stops this block.

        # ====================================================



        if recording:



            if video_writer is not None:



                video_writer.write(

                    camera_frame

                )





            if (



                recording_end_time is not None



                and



                current_time

                >= recording_end_time



            ):



                print()



                print(

                    "⏹ Recording finished."

                )





                recording_success = (

                    finalize_recording(



                        video_writer,



                        current_raw_filepath,



                        current_final_filepath



                    )

                )





                video_writer = None



                recording = False



                recording_end_time = None





                # ------------------------------------------------

                # SEND ALERT AFTER VIDEO IS SAVED

                # ------------------------------------------------



                if recording_success:



                    send_emergency_alert(

                        current_video_filename

                    )





                current_video_filename = None



                current_raw_filepath = None



                current_final_filepath = None





        # ====================================================
        # THREE-COLUMN SAFE SIGHT GUARDIAN UI
        # LEFT: PHASE 2 | CENTER: CAMERA | RIGHT: PHASE 1
        # ====================================================

        UI_W, UI_H = 1500, 760
        frame = np.full((UI_H, UI_W, 3), BG, dtype=np.uint8)

        # Header
        text(frame, "SAFE SIGHT GUARDIAN", (30, 45), 0.75, CYAN, 2)
        text(frame, "EMERGENCY AI", (365, 43), 0.55, WHITE, 1)
        cv2.circle(frame, (1310, 32), 7, GREEN, -1)
        text(frame, "SYSTEM ONLINE", (1330, 38), 0.5, GREEN, 1)
        cv2.line(frame, (30, 65), (1470, 65), (50, 60, 70), 1)

        # Three panels: Phase 2 left, camera center, Phase 1 right
        panel(frame, 30, 90, 250, 690)
        panel(frame, 270, 90, 1170, 690)
        panel(frame, 1190, 90, 1470, 690)

        # Camera feed in the center
        frame[90:690, 270:1170] = camera_frame
        cv2.rectangle(frame, (270, 90), (1170, 690), CYAN, 1)

        # ====================================================
        # LEFT PANEL: PHASE 2 ACTION DETECTION
        # ====================================================

        text(frame, "PHASE 2", (48, 125), 0.65, CYAN, 2)
        text(frame, "ACTION DETECTION", (48, 153), 0.42, WHITE, 1)
        cv2.line(frame, (45, 170), (235, 170), (60, 65, 75), 1)

        phase2_state = str(phase2_status_cache.get("state", "STARTING")).upper()
        try:
            phase2_score = float(phase2_status_cache.get("activity_score", phase2_status_cache.get("max_score", 0.0))) * 100.0
        except (TypeError, ValueError):
            phase2_score = 0.0

        phase2_message = str(phase2_status_cache.get("message", "Waiting for worker"))
        if not phase2_enabled or phase2_state == "OFFLINE":
            phase2_color, phase2_label = RED, "OFFLINE"
        elif phase2_state == "ERROR":
            phase2_color, phase2_label = RED, "ERROR"
        elif phase2_state == "CANDIDATE":
            phase2_color, phase2_label = ORANGE, "REVIEW CANDIDATE"
        elif phase2_state == "PROCESSING":
            phase2_color, phase2_label = ORANGE, "ANALYZING"
        elif phase2_state in ("STARTING", "ONLINE"):
            phase2_color, phase2_label = ORANGE, phase2_state
        else:
            phase2_color, phase2_label = GREEN, "MONITORING"

        cv2.circle(frame, (53, 198), 5, phase2_color, -1)
        text(frame, phase2_label, (66, 204), 0.40, phase2_color, 1)

        text(frame, "STABLE ACTIVITY", (48, 247), 0.40, GRAY, 1)
        text(frame, f"{phase2_score:.1f}%", (48, 278), 0.85, WHITE, 2)
        cv2.rectangle(frame, (48, 295), (228, 307), (50, 55, 65), -1)
        score_width = int(180 * min(max(phase2_score / 100.0, 0.0), 1.0))
        if score_width:
            cv2.rectangle(frame, (48, 295), (48 + score_width, 307), phase2_color, -1)
        try:
            raw_clip_percent = float(phase2_status_cache.get("decision_score", phase2_status_cache.get("clip_score", 0.0))) * 100.0
        except (TypeError, ValueError):
            raw_clip_percent = 0.0
        text(frame, f"CURRENT CLIP: {raw_clip_percent:.1f}%", (48, 329), 0.34, WHITE, 1)

        save_status = str(phase2_status_cache.get("candidate_save_status", "NOT_TRIGGERED")).upper()
        save_color = GREEN if save_status == "SAVED" else (RED if save_status == "FAILED" else GRAY)
        text(frame, f"VIDEO SAVE: {save_status[:12]}", (48, 345), 0.34, save_color, 1)
        text(frame, "WORKER STATUS", (48, 378), 0.40, GRAY, 1)
        text(frame, phase2_state[:22], (48, 405), 0.40, phase2_color, 1)
        text(frame, "LATEST RESULT", (48, 445), 0.40, GRAY, 1)

        # Wrap the result message to fit inside the narrow Phase 2 panel.
        message_lines = []
        current_line = ""
        for word in phase2_message.split():
            candidate_line = (current_line + " " + word).strip()
            if len(candidate_line) > 23:
                if current_line:
                    message_lines.append(current_line)
                current_line = word
            else:
                current_line = candidate_line
        if current_line:
            message_lines.append(current_line)
        for index, message_line in enumerate(message_lines[:5]):
            text(frame, message_line, (48, 473 + index * 20), 0.32, WHITE, 1)

        if phase2_state == "CANDIDATE":
            text(frame, "HUMAN REVIEW", (48, 590), 0.38, ORANGE, 1)
            text(frame, "REQUIRED", (48, 613), 0.38, ORANGE, 1)
        text(frame, "VIDEO ANALYSIS", (48, 660), 0.34, GRAY, 1)

        # ====================================================
        # RIGHT PANEL: PHASE 1 HAND-SIGN DETECTION
        # ====================================================

        text(frame, "PHASE 1", (1210, 125), 0.65, CYAN, 2)
        text(frame, "HAND SIGN", (1210, 153), 0.42, WHITE, 1)
        cv2.line(frame, (1205, 170), (1455, 170), (60, 65, 75), 1)

        if emergency_verified:
            status, status_color = "EMERGENCY", RED
        elif sequence_state == SEQUENCE_SIGN1_DETECTED and emergency_start_time is not None:
            status, status_color = "VERIFYING SIGN 2", ORANGE
        elif sequence_state == SEQUENCE_SIGN1_DETECTED:
            status, status_color = "SIGN 1 DETECTED", ORANGE
        elif sign1_detected:
            status, status_color = "SIGN 1", ORANGE
        else:
            status, status_color = "SAFE", GREEN

        text(frame, "DETECTION STATUS", (1210, 205), 0.38, GRAY, 1)
        text(frame, status, (1210, 237), 0.54, status_color, 2)
        text(frame, "MODEL CONFIDENCE", (1210, 285), 0.38, GRAY, 1)
        text(frame, f"{probability * 100:.1f}%", (1210, 319), 0.75, WHITE, 2)

        cv2.rectangle(frame, (1210, 336), (1445, 348), (50, 55, 65), -1)
        confidence_width = int(235 * min(max(probability, 0.0), 1.0))
        if confidence_width:
            cv2.rectangle(frame, (1210, 336), (1210 + confidence_width, 348), status_color, -1)

        text(frame, "SEQUENCE", (1210, 390), 0.42, GRAY, 1)
        if emergency_verified:
            sequence_label, sequence_color = "EMERGENCY VERIFIED", RED
        elif sequence_state == SEQUENCE_SIGN1_DETECTED:
            sequence_label, sequence_color = "WAITING FOR SIGN 2", ORANGE
        elif sign1_detected:
            sequence_label, sequence_color = "SIGN 1 DETECTED", ORANGE
        else:
            sequence_label, sequence_color = "WAITING FOR SIGN 1", WHITE
        text(frame, sequence_label, (1210, 420), 0.38, sequence_color, 1)

        if sequence_state == SEQUENCE_WAITING:
            text(frame, "Open 4 + thumb folded", (1210, 450), 0.32, GRAY, 1)
        elif emergency_verified:
            text(frame, "Sequence completed", (1210, 450), 0.34, RED, 1)
        else:
            text(frame, "All fingers closed for sign 2", (1210, 450), 0.30, GRAY, 1)

        text(frame, "SYSTEM STATUS", (1210, 505), 0.42, GRAY, 1)
        phase1_systems = [
            ("CAMERA", "ONLINE", GREEN),
            ("AI MODEL", "ONLINE", GREEN),
            ("HAND AI", "ONLINE" if hand_detected else "READY", GREEN),
            ("SEQUENCE", "ACTIVE" if sequence_state == SEQUENCE_SIGN1_DETECTED else "READY",
             ORANGE if sequence_state == SEQUENCE_SIGN1_DETECTED else GREEN),
        ]
        for index, (name, value, color) in enumerate(phase1_systems):
            y = 535 + index * 27
            cv2.circle(frame, (1217, y), 4, color, -1)
            text(frame, name, (1230, y + 5), 0.34, WHITE, 1)
            text(frame, value, (1350, y + 5), 0.32, color, 1)

        if recording:
            text(frame, "RECORDING VIDEO", (1210, 660), 0.38, RED, 2)
        elif emergency_verified:
            text(frame, "ALERT TRIGGERED", (1210, 660), 0.38, RED, 2)
        else:
            text(frame, "RECORDING STANDBY", (1210, 660), 0.32, GRAY, 1)

        # Footer
        text(frame, "LEFT: PHASE 2 ACTION ANALYSIS", (30, 730), 0.38, CYAN, 1)
        text(frame, "CENTER: LIVE CAMERA", (600, 730), 0.38, GRAY, 1)
        text(frame, "RIGHT: PHASE 1 HAND SIGNS", (1190, 730), 0.38, CYAN, 1)


        # ====================================================

        # SHOW WINDOW

        # ====================================================



        cv2.imshow(

            WINDOW_NAME,

            frame

        )





        # ====================================================

        # KEYBOARD

        # ====================================================



        key = cv2.waitKey(1) & 0xFF





        if (



            key == ord("q")



            or key == ord("Q")



            or key == 27



        ):



            print()



            print(

                "🛑 Exit requested."

            )



            break





# ============================================================

# CTRL+C HANDLING

# ============================================================



except KeyboardInterrupt:



    print()



    print(

        "🛑 Ctrl+C detected."

    )





except Exception as e:



    print()



    print(

        "❌ Unexpected error:"

    )



    print(e)



    traceback.print_exc()





# ============================================================

# CLEANUP

# ============================================================



finally:



    print()



    print(

        "Cleaning up..."

    )





    # --------------------------------------------------------

    # RELEASE VIDEO WRITER

    # --------------------------------------------------------



    if video_writer is not None:



        video_writer.release()



        video_writer = None





    # --------------------------------------------------------

    # --------------------------------------------------------
    # STOP PHASE 2 WORKER (does not own the webcam)
    # --------------------------------------------------------
    if phase2_worker_process is not None and phase2_worker_process.poll() is None:
        phase2_worker_process.terminate()
        try:
            phase2_worker_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            phase2_worker_process.kill()
            phase2_worker_process.wait(timeout=2)
    # --------------------------------------------------------
    # RELEASE CAMERA
    # --------------------------------------------------------
    cap.release()





    # --------------------------------------------------------

    # CLOSE WINDOWS

    # --------------------------------------------------------



    cv2.destroyAllWindows()





    # --------------------------------------------------------

    # CLOSE MEDIAPIPE

    # --------------------------------------------------------



    hand_landmarker.close()





    print(

        "Camera released."

    )



    print(

        "OpenCV windows closed."

    )



    print(

        "MediaPipe closed."

    )



    print()



    print(

        "System stopped."

    )
