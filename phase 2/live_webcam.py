
import cv2
import os
import sys
import time
import json
import tempfile
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np

# ---------- PATHS ----------
PHASE2_DIR = Path(__file__).resolve().parent
VDMIL_DIR = PHASE2_DIR / "vdmil"
CLIP_DIR = PHASE2_DIR / "candidate_clips"
CLIP_DIR.mkdir(parents=True, exist_ok=True)
EVENT_LOG = CLIP_DIR / "events.jsonl"

sys.path.insert(0, str(VDMIL_DIR))

from movinet_classifier import MovinetClassifier
from activity_detection import run_video_processing


# ---------- SETTINGS ----------
CAMERA_ID = 0
TARGET_FPS = 8
WINDOW_FRAMES = 8
STRIDE = 4

# Model inference buffer
BUFFER_SECONDS = 8
BUFFER_SIZE = TARGET_FPS * BUFFER_SECONDS
INFERENCE_INTERVAL = 4.0

# Experimental thresholds - validate using labelled videos
CANDIDATE_THRESHOLD = 0.70
MIN_CONSECUTIVE_HIGH = 3

# Candidate clip recording
PRE_EVENT_SECONDS = 5
POST_EVENT_SECONDS = 5
COOLDOWN_SECONDS = 60

PRE_EVENT_FRAMES = PRE_EVENT_SECONDS * TARGET_FPS
POST_EVENT_FRAMES = POST_EVENT_SECONDS * TARGET_FPS

MOVINET_DIR = VDMIL_DIR / "movinet_weights"
CHECKPOINT = VDMIL_DIR / "test" / "model_20.pt"


# ---------- SHARED INFERENCE STATE ----------
frame_buffer = deque(maxlen=BUFFER_SIZE)

state = {
    "busy": False,
    "score": None,
    "mean_score": None,
    "high_run": 0,
    "error": None,
    "last_finished": None,
}

lock = threading.Lock()


def analyze_video(frames):
    """Analyze a snapshot of webcam frames using VD-MIL."""
    temp_path = None

    try:
        fd, temp_path = tempfile.mkstemp(
            suffix=".mp4",
            dir=str(PHASE2_DIR),
        )
        os.close(fd)

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            temp_path,
            fourcc,
            TARGET_FPS,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError("Could not create temporary video.")

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

        scores = run_video_processing(
            temp_path,
            classifier,
            target_fps=TARGET_FPS,
            length_window=WINDOW_FRAMES,
            stride=STRIDE,
        )

        scores = np.asarray(scores, dtype=float).reshape(-1)
        scores = scores[np.isfinite(scores)]

        if scores.size == 0:
            raise RuntimeError("VD-MIL returned no valid scores.")

        # Longest consecutive run within this analysis result.
        above_threshold = scores >= CANDIDATE_THRESHOLD
        longest_run = 0
        current_run = 0

        for is_high in above_threshold:
            if is_high:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0

        with lock:
            state["score"] = float(np.max(scores))
            state["mean_score"] = float(np.mean(scores))
            state["high_run"] = longest_run
            state["error"] = None
            state["last_finished"] = time.time()

    except Exception as exc:
        with lock:
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["high_run"] = 0
            state["last_finished"] = time.time()

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

        with lock:
            state["busy"] = False


def save_event_log(event_info, clip_path, frames_written):
    """Append one completed or interrupted recording to JSONL."""
    entry = {
        **event_info,
        "clip_path": str(clip_path.resolve()),
        "frames_written": frames_written,
        "review_status": "pending_human_review",
    }

    with open(EVENT_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry) + "\n")

    print(f"Candidate clip saved: {clip_path}")
    print(f"Event log updated: {EVENT_LOG}")


def main():
    global classifier

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing classifier checkpoint: {CHECKPOINT}"
        )

    if not MOVINET_DIR.exists():
        raise FileNotFoundError(
            f"Missing MoViNet weights folder: {MOVINET_DIR}"
        )

    print("Loading pretrained VD-MIL on CPU...")
    start = time.perf_counter()

    classifier = MovinetClassifier(
        str(MOVINET_DIR),
        [str(CHECKPOINT)],
        device="cpu",
    )

    print(f"Model loaded in {time.perf_counter() - start:.2f} seconds.")
    print("Opening webcam...")

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open webcam. Close other camera applications and retry."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Separate buffer for recording video before a candidate trigger.
    pre_event_buffer = deque(maxlen=PRE_EVENT_FRAMES)

    recording_writer = None
    recording_path = None
    recording_event = None
    recording_remaining = 0
    recording_frames_written = 0

    last_candidate_time = 0.0
    last_handled_finished = None
    last_sample = 0.0
    last_inference_start = 0.0

    fps_start = time.perf_counter()
    displayed_frames = 0
    display_fps = 0.0

    print("Live detector running.")
    print(f"Candidate clips folder: {CLIP_DIR}")
    print("Press Q while the webcam window is focused to exit.")
    print("All candidate clips require human review.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            now = time.perf_counter()
            displayed_frames += 1

            # Sample frames at approximately TARGET_FPS.
            sampled = now - last_sample >= 1.0 / TARGET_FPS

            if sampled:
                sample = frame.copy()
                last_sample = now

                # Keep recent frames for pre-event recording.
                pre_event_buffer.append(sample)

                with lock:
                    frame_buffer.append(sample.copy())

                # Write post-trigger frames if a clip is being recorded.
                if recording_writer is not None:
                    recording_writer.write(sample)
                    recording_frames_written += 1
                    recording_remaining -= 1

                    if recording_remaining <= 0:
                        recording_writer.release()
                        recording_writer = None

                        save_event_log(
                            recording_event,
                            recording_path,
                            recording_frames_written,
                        )

                        recording_path = None
                        recording_event = None
                        recording_frames_written = 0

            # Start inference without blocking webcam capture.
            with lock:
                can_infer = (
                    len(frame_buffer) == BUFFER_SIZE
                    and not state["busy"]
                    and now - last_inference_start >= INFERENCE_INTERVAL
                )

                if can_infer:
                    snapshot = list(frame_buffer)
                    state["busy"] = True

            if can_infer:
                last_inference_start = now

                threading.Thread(
                    target=analyze_video,
                    args=(snapshot,),
                    daemon=True,
                ).start()

            with lock:
                score = state["score"]
                mean_score = state["mean_score"]
                high_run = state["high_run"]
                error = state["error"]
                finished = state["last_finished"]
                busy = state["busy"]

            # Only consider each newly completed inference once.
            new_result = (
                finished is not None
                and finished != last_handled_finished
            )

            if new_result:
                last_handled_finished = finished

                candidate_detected = (
                    error is None
                    and score is not None
                    and score >= CANDIDATE_THRESHOLD
                    and high_run >= MIN_CONSECUTIVE_HIGH
                )

                cooldown_finished = (
                    time.time() - last_candidate_time >= COOLDOWN_SECONDS
                )

                if (
                    candidate_detected
                    and cooldown_finished
                    and recording_writer is None
                    and len(pre_event_buffer) > 0
                ):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    recording_path = (
                        CLIP_DIR / f"candidate_{timestamp}.mp4"
                    )

                    height, width = pre_event_buffer[-1].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    recording_writer = cv2.VideoWriter(
                        str(recording_path),
                        fourcc,
                        TARGET_FPS,
                        (width, height),
                    )

                    if recording_writer.isOpened():
                        recording_event = {
                            "timestamp": datetime.now().isoformat(
                                timespec="seconds"
                            ),
                            "max_score": float(score),
                            "mean_score": (
                                float(mean_score)
                                if mean_score is not None
                                else None
                            ),
                            "consecutive_high_windows": int(high_run),
                            "threshold": CANDIDATE_THRESHOLD,
                            "pre_event_seconds": PRE_EVENT_SECONDS,
                            "post_event_seconds": POST_EVENT_SECONDS,
                            "note": (
                                "Experimental model candidate; "
                                "not a verified incident."
                            ),
                        }

                        # Write the pre-event frames first.
                        pre_frames = list(pre_event_buffer)
                        for old_frame in pre_frames:
                            recording_writer.write(old_frame)

                        recording_frames_written = len(pre_frames)
                        recording_remaining = POST_EVENT_FRAMES
                        last_candidate_time = time.time()

                        print(
                            f"CANDIDATE FLAGGED | score={score:.3f} | "
                            f"recording={recording_path.name}"
                        )
                    else:
                        recording_writer.release()
                        recording_writer = None
                        recording_path = None
                        recording_event = None
                        print("Could not create candidate recording.")

            # Measure webcam display FPS.
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                display_fps = displayed_frames / elapsed
                displayed_frames = 0
                fps_start = time.perf_counter()

            # On-screen information.
            cv2.putText(
                frame,
                "Safe Sight Guardian Phase 2 - VD-MIL Experimental",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

            if score is None:
                score_text = "Max window score: waiting"
                status = (
                    "Collecting video..."
                    if len(frame_buffer) < BUFFER_SIZE
                    else "Analyzing..."
                )
            else:
                score_text = f"Max window score: {score:.3f}"

                if finished is not None and time.time() - finished > 15:
                    status = "Score stale - waiting for inference"
                elif high_run >= MIN_CONSECUTIVE_HIGH:
                    status = "CANDIDATE - HUMAN REVIEW REQUIRED"
                elif score >= CANDIDATE_THRESHOLD:
                    status = "High-score spike - not yet a candidate"
                else:
                    status = "No persistent high-score pattern"

            cv2.putText(
                frame,
                score_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                status,
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 165, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Camera FPS: {display_fps:.1f} | "
                f"Inference: {'busy' if busy else 'idle'}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
            )

            if mean_score is not None:
                cv2.putText(
                    frame,
                    f"Mean window score: {mean_score:.3f}",
                    (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                f"Consecutive high windows: {high_run}",
                (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
            )

            recording_status = (
                "RECORDING CANDIDATE CLIP"
                if recording_writer is not None
                else "Recording: idle"
            )

            cv2.putText(
                frame,
                recording_status,
                (10, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255) if recording_writer else (255, 255, 255),
                2,
            )

            if error:
                cv2.putText(
                    frame,
                    f"Error: {error[:65]}",
                    (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Safe Sight Guardian Phase 2 - VD-MIL", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Finalize a partially recorded clip if the user exits early.
        if recording_writer is not None:
            recording_writer.release()

            if recording_path is not None and recording_event is not None:
                save_event_log(
                    recording_event,
                    recording_path,
                    recording_frames_written,
                )

        print("Webcam stopped.")


if __name__ == "__main__":
    main()
