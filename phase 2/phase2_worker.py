"""Safe Sight Guardian Phase 2 VD-MIL worker.

Consumes clips created by Phase 1 from its existing webcam frames. It never
opens a webcam. Candidate clips are saved for human review, not treated as
confirmed violence.
"""
import glob
import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VDMIL_DIR = os.path.join(BASE_DIR, "vdmil")
INCOMING_DIR = os.path.join(BASE_DIR, "incoming_clips")
CANDIDATE_DIR = os.path.join(BASE_DIR, "candidate_clips")
BROWSER_READY_DIR = os.path.join(CANDIDATE_DIR, "browser_ready")
EVENTS_FILE = os.path.join(CANDIDATE_DIR, "events.jsonl")
STATUS_FILE = os.path.join(CANDIDATE_DIR, "status.json")
MODEL_FOLDER = os.path.join(VDMIL_DIR, "movinet_weights")
DETECTOR_MODEL = os.path.join(VDMIL_DIR, "test", "model_20.pt")

TARGET_FPS = 8
LENGTH_WINDOW = 8
STRIDE = 4
CANDIDATE_THRESHOLD = 0.70
# Detection uses the current clip score, NOT the smoothed display score.
# Require two consecutive overlapping clips above threshold to reduce spikes
# while avoiding the previous 3-window + EMA delay.
MIN_CONSECUTIVE_HIGH_CLIPS = 2
POLL_SECONDS = 0.5
DISPLAY_EMA_ALPHA = 0.30

os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(CANDIDATE_DIR, exist_ok=True)
os.makedirs(BROWSER_READY_DIR, exist_ok=True)


def create_browser_ready_copy(source_path, output_path):
    """Create an H.264 copy for browser playback; retain the original model clip."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    temp_path = output_path + ".tmp.mp4"
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", source_path, "-map", "0:v:0", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", temp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=90,
        )
        if not os.path.isfile(temp_path) or os.path.getsize(temp_path) == 0:
            raise IOError("FFmpeg created an empty browser video")
        os.replace(temp_path, output_path)
        return True
    except Exception as exc:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        print("Could not create browser-compatible candidate video:", exc, flush=True)
        return False


def write_status(**values):
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **values,
    }
    temp_path = STATUS_FILE + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(temp_path, STATUS_FILE)
    except OSError as exc:
        print("Could not write Phase 2 status:", exc, flush=True)


def as_unit_score(value):
    """Normalize common model output scales to 0..1 for display/thresholding.

    If outputs are already probabilities/scores in 0..1 they are unchanged.
    If they are percentages in 0..100 they are converted. Other score ranges
    are clipped, so inspect model semantics before interpreting as probability.
    """
    value = float(value)
    if 1.0 < value <= 100.0:
        value /= 100.0
    return float(np.clip(value, 0.0, 1.0))


def main():
    import sys
    if VDMIL_DIR not in sys.path:
        sys.path.insert(0, VDMIL_DIR)

    from movinet_classifier import MovinetClassifier
    from activity_detection import run_video_processing
    try:
        from signal_smoothing import smooth_scores
    except Exception:
        smooth_scores = None

    write_status(state="STARTING", activity_score=0.0, max_score=0.0,
                 raw_max_score=0.0, clip_score=0.0, mean_score=0.0,
                 message="Loading VD-MIL model")
    print("Safe Sight Guardian Phase 2 worker starting on CPU.", flush=True)
    print("Model folder:", MODEL_FOLDER, flush=True)
    classifier = MovinetClassifier(MODEL_FOLDER, [DETECTOR_MODEL], device="cpu")

    stable_score = None
    previous_high_clips = 0
    write_status(state="ONLINE", activity_score=0.0, max_score=0.0,
                 raw_max_score=0.0, clip_score=0.0, mean_score=0.0,
                 consecutive_high_clips=0,
                 message="Waiting for analysis window")
    print("VD-MIL loaded. Waiting for clips; no webcam will be opened.", flush=True)

    while True:
        files = sorted(glob.glob(os.path.join(INCOMING_DIR, "analysis_*.mp4")))
        if not files:
            time.sleep(POLL_SECONDS)
            continue

        clip_path = files[0]
        write_status(
            state="PROCESSING",
            activity_score=round(float(stable_score or 0.0), 4),
            max_score=round(float(stable_score or 0.0), 4),
            consecutive_high_clips=previous_high_clips,
            message=f"Analyzing {os.path.basename(clip_path)}",
        )
        try:
            scores = run_video_processing(
                clip_path, classifier, TARGET_FPS, LENGTH_WINDOW, STRIDE
            )
            raw_values = np.asarray(scores, dtype=float).reshape(-1)
            raw_values = raw_values[np.isfinite(raw_values)]
            if raw_values.size == 0:
                previous_high_clips = 0
                write_status(state="MONITORING", message="Model returned no usable scores",
                             consecutive_high_clips=0)
                print("VD-MIL returned no usable scores for", os.path.basename(clip_path), flush=True)
                continue

            # Preserve original values for diagnostics; normalize scores for
            # consistent display and threshold comparison.
            normalized_raw = np.asarray([as_unit_score(v) for v in raw_values], dtype=float)
            analysed_values = normalized_raw
            if smooth_scores is not None:
                try:
                    smoothed = np.asarray(smooth_scores(normalized_raw), dtype=float).reshape(-1)
                    smoothed = smoothed[np.isfinite(smoothed)]
                    if smoothed.size:
                        analysed_values = np.asarray([as_unit_score(v) for v in smoothed], dtype=float)
                except Exception as smooth_exc:
                    print("Optional signal smoothing failed; using raw scores:", smooth_exc, flush=True)
                    analysed_values = normalized_raw

            raw_max_score = float(np.max(normalized_raw))
            raw_mean_score = float(np.mean(normalized_raw))
            # A robust clip score avoids letting one isolated spike dominate.
            clip_score = float(np.percentile(analysed_values, 75))
            mean_score = float(np.mean(analysed_values))

            # UI-only EMA. This value must NEVER gate candidate detection.
            if stable_score is None:
                stable_score = clip_score
            else:
                stable_score = (DISPLAY_EMA_ALPHA * clip_score
                                + (1.0 - DISPLAY_EMA_ALPHA) * stable_score)
            activity_score = float(np.clip(stable_score, 0.0, 1.0))

            # Use current clip evidence for detection, independent of the EMA.
            # If a clip's robust score reaches the threshold, count it as high.
            if clip_score >= CANDIDATE_THRESHOLD:
                previous_high_clips += 1
            else:
                previous_high_clips = 0
            is_candidate = previous_high_clips >= MIN_CONSECUTIVE_HIGH_CLIPS

            write_status(
                state="CANDIDATE" if is_candidate else "MONITORING",
                activity_score=round(activity_score, 4),
                max_score=round(activity_score, 4),  # old UI compatibility
                raw_max_score=round(raw_max_score, 4),
                raw_mean_score=round(raw_mean_score, 4),
                clip_score=round(clip_score, 4),
                mean_score=round(mean_score, 4),
                decision_score=round(clip_score, 4),
                consecutive_high_clips=int(previous_high_clips),
                threshold=CANDIDATE_THRESHOLD,
                candidate_save_status=("PENDING_SAVE" if is_candidate else "NOT_TRIGGERED"),
                candidate_file="",
                candidate_bytes=0,
                message=("Candidate threshold met; saving review clip"
                         if is_candidate else "No candidate in latest window; no review clip saved"),
            )
            print(
                f"VD-MIL raw_max={raw_max_score:.3f}, clip_p75={clip_score:.3f}, "
                f"stable_display={activity_score:.3f}, mean={mean_score:.3f}, "
                f"high_clip_streak={previous_high_clips}", flush=True,
            )

            if is_candidate:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                candidate_name = f"candidate_{stamp}.mp4"
                candidate_path = os.path.join(CANDIDATE_DIR, candidate_name)
                try:
                    shutil.copy2(clip_path, candidate_path)
                    candidate_saved = os.path.isfile(candidate_path) and os.path.getsize(candidate_path) > 0
                    candidate_bytes = os.path.getsize(candidate_path) if candidate_saved else 0
                    if not candidate_saved:
                        raise IOError("Candidate clip is missing or empty after copy")
                except Exception as save_exc:
                    write_status(
                        state="SAVE_ERROR",
                        activity_score=round(activity_score, 4),
                        decision_score=round(clip_score, 4),
                        clip_score=round(clip_score, 4),
                        threshold=CANDIDATE_THRESHOLD,
                        consecutive_high_clips=int(previous_high_clips),
                        candidate_save_status="FAILED",
                        candidate_file=os.path.abspath(candidate_path),
                        candidate_bytes=0,
                        message=f"Candidate detected but video save FAILED: {save_exc}",
                    )
                    print(
                        f"PHASE 2 SAVE FAILED: score={clip_score:.3f}; path={candidate_path}; error={save_exc}",
                        flush=True,
                    )
                    previous_high_clips = 0
                    continue

                print(
                    f"PHASE 2 VIDEO SAVE VERIFIED: exists={candidate_saved}, bytes={candidate_bytes}, path={os.path.abspath(candidate_path)}",
                    flush=True,
                )
                browser_video_saved = create_browser_ready_copy(
                    candidate_path, os.path.join(BROWSER_READY_DIR, candidate_name)
                )
                if browser_video_saved:
                    print("Browser-ready H.264 candidate saved.", flush=True)
                write_status(
                    state="CANDIDATE",
                    activity_score=round(activity_score, 4),
                    decision_score=round(clip_score, 4),
                    clip_score=round(clip_score, 4),
                    threshold=CANDIDATE_THRESHOLD,
                    consecutive_high_clips=int(previous_high_clips),
                    candidate_save_status="SAVED",
                    candidate_file=os.path.abspath(candidate_path),
                    candidate_bytes=int(candidate_bytes),
                    browser_video_status="READY" if browser_video_saved else "RAW_FORMAT",
                    candidate_saved_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    message="Candidate detected; video saved and verified for human review",
                )
                event = {
                    "event_type": "PHASE2_CANDIDATE",
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "activity_score": round(activity_score, 4),
                    "decision_score": round(clip_score, 4),
                    "clip_score": round(clip_score, 4),
                    "raw_max_score": round(raw_max_score, 4),
                    "raw_mean_score": round(raw_mean_score, 4),
                    "mean_score": round(mean_score, 4),
                    "consecutive_high_clips": int(previous_high_clips),
                    "threshold": CANDIDATE_THRESHOLD,
                    "clip_path": os.path.abspath(candidate_path),
                    "clip_filename": candidate_name,
                    "browser_clip_filename": candidate_name if browser_video_saved else None,
                    "review_status": "pending_human_review",
                    "note": "Automated model candidate only; human review required. Score is not a calibrated probability of violence.",
                }
                with open(EVENTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
                print("PHASE 2 CANDIDATE SAVED FOR HUMAN REVIEW:", candidate_path, flush=True)
                # Reset streak after saving so a sustained event does not save
                # every subsequent clip as a new candidate.
                previous_high_clips = 0
        except Exception as exc:
            write_status(state="ERROR", message=str(exc))
            print("Phase 2 processing error:", exc, flush=True)
            traceback.print_exc()
        finally:
            try:
                if os.path.exists(clip_path):
                    os.remove(clip_path)
            except OSError:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Phase 2 worker stopped.", flush=True)
    except Exception as exc:
        try:
            write_status(state="ERROR", message=str(exc))
        except Exception:
            pass
        print("Phase 2 worker could not start:", exc, flush=True)
        traceback.print_exc()
