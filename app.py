import base64
import logging
import os
import threading
import time
from collections import deque
from typing import Dict, List

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from ultralytics import YOLO

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config (all overridable via environment variables) ────────────────────────
APP_DIR            = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH         = os.environ.get("MODEL_PATH",         os.path.join(APP_DIR, "models", "best.pt"))
CAMERA_SOURCE      = int(os.environ.get("CAMERA_SOURCE",      "0"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
IMG_SIZE           = int(os.environ.get("IMG_SIZE",           "640"))
ZONE_NAME          = os.environ.get("ZONE_NAME",          "Main Entrance")
DEBUG_MODE         = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
MAX_UPLOAD_MB      = int(os.environ.get("MAX_UPLOAD_MB",      "16"))    # image upload limit
MAX_VIDEO_MB       = int(os.environ.get("MAX_VIDEO_MB",       "500"))   # video upload limit (videos are large)
EVENT_THROTTLE_S   = float(os.environ.get("EVENT_THROTTLE_S", "0.6"))  # seconds between log entries
STREAM_FPS         = int(os.environ.get("STREAM_FPS",         "20"))    # video-feed target FPS

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

VIDEO_UPLOAD_DIR = os.path.join(APP_DIR, "uploaded_videos")
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=APP_DIR, static_url_path="")
# Set limit high enough for video uploads; image size is enforced in code
app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_MB * 1024 * 1024


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_label(raw_label: str) -> str:
    label = str(raw_label).strip().lower().replace("_", " ").replace("-", " ")
    if "no" in label and "helmet" in label:
        return "No Helmet"
    if "without" in label and "helmet" in label:
        return "No Helmet"
    if "helmet" in label:
        return "Helmet"
    return str(raw_label).strip().title()


def make_placeholder_frame(message: str) -> bytes:
    frame = np.full((480, 760, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (18, 18), (742, 462), (223, 240, 248), 3)
    cv2.putText(frame, "Industrial Safety Helmet Detection", (44, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (61, 126, 166), 2, cv2.LINE_AA)
    cv2.putText(frame, message, (44, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 82, 104), 2, cv2.LINE_AA)
    cv2.putText(frame, "Place your trained model at models/best.pt", (44, 295),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (60, 82, 104), 2, cv2.LINE_AA)
    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else b""


def _allowed_image(filename: str) -> bool:
    """Return True only for image file extensions the backend can safely decode."""
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_IMAGE_EXTENSIONS


def _allowed_video(filename: str) -> bool:
    """Return True only for video file extensions OpenCV can read."""
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_VIDEO_EXTENSIONS


# ── Backend ───────────────────────────────────────────────────────────────────

class HelmetDetectionBackend:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.camera_source = CAMERA_SOURCE
        self.cap = None
        self.thread = None
        self.latest_frame: bytes = make_placeholder_frame("Backend idle. Click Start Detection.")
        self.recent_events: deque[Dict] = deque(maxlen=12)
        self.stats: Dict[str, int] = {
            "total": 0,
            "helmet": 0,
            "no_helmet": 0,
            "other": 0,
        }
        self.last_event_at: float = 0.0
        self.model_error: str = ""
        self.model = None
        self.model_loaded: bool = False
        self.is_video_file: bool = False       # True when source is a file, not a webcam
        self.video_filename: str = ""          # display name shown in stats
        self.load_model()

    # ── Model ─────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        if not os.path.exists(MODEL_PATH):
            self.model_error = f"Model file not found at {MODEL_PATH}"
            self.model_loaded = False
            log.warning(self.model_error)
            return
        try:
            self.model = YOLO(MODEL_PATH)
            self.model_loaded = True
            self.model_error = ""
            log.info("YOLO model loaded from %s", MODEL_PATH)
        except Exception as exc:
            self.model = None
            self.model_loaded = False
            self.model_error = str(exc)
            log.error("Failed to load model: %s", exc)

    # ── Camera lifecycle ──────────────────────────────────────────────────────

    def start(self, source=None) -> Dict:
        """Start detection. source = None → webcam (CAMERA_SOURCE), or a video file path."""
        with self.lock:
            if not self.model_loaded:
                self.load_model()
            if not self.model_loaded:
                self.latest_frame = make_placeholder_frame("Model not loaded. Check models/best.pt")
                return {"ok": False, "message": self.model_error or "Model could not be loaded."}
            if self.running:
                return {"ok": True, "message": "Detection is already running."}

            # Determine source
            if source is None:
                actual_source = CAMERA_SOURCE
                self.is_video_file = False
                self.video_filename = ""
            else:
                actual_source = source
                self.is_video_file = True
                self.video_filename = os.path.basename(source)

            self.cap = cv2.VideoCapture(actual_source)
            if not self.cap.isOpened():
                self.cap = None
                msg = (
                    f"Could not open video file: {self.video_filename}"
                    if self.is_video_file
                    else "Could not open webcam. Check if another app is using it."
                )
                self.latest_frame = make_placeholder_frame(msg)
                return {"ok": False, "message": msg}

            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            log.info("Detection started (source: %s)", actual_source)
            return {
                "ok": True,
                "message": (
                    f"Video detection started: {self.video_filename}"
                    if self.is_video_file
                    else "Webcam detection started successfully."
                ),
            }

    def stop(self) -> Dict:
        with self.lock:
            self.running = False
            cap = self.cap
            self.cap = None

        if cap is not None:
            cap.release()

        self.latest_frame = make_placeholder_frame("Detection stopped. Click Start Detection.")
        log.info("Detection stopped.")
        return {"ok": True, "message": "Detection stopped successfully."}

    def reset_stats(self) -> Dict:
        """Zero out counters and clear the event log without stopping the camera."""
        with self.lock:
            self.stats = {"total": 0, "helmet": 0, "no_helmet": 0, "other": 0}
            self.recent_events.clear()
            self.last_event_at = 0.0
        log.info("Stats reset by user.")
        return {"ok": True, "message": "Stats reset successfully."}

    # ── Inference loop ────────────────────────────────────────────────────────

    def _add_event(self, label: str, confidence: float) -> None:
        event = {
            "time": time.strftime("%H:%M:%S"),
            "zone": ZONE_NAME,
            "status": label,
            "confidence": int(round(confidence * 100)),
        }
        self.recent_events.appendleft(event)

    def _update_stats_from_boxes(self, detections: List[Dict]) -> None:
        for det in detections:
            label = det["label"]
            if label == "Helmet":
                self.stats["helmet"] += 1
            elif label == "No Helmet":
                self.stats["no_helmet"] += 1
            else:
                self.stats["other"] += 1
            self.stats["total"] += 1

    def _loop(self) -> None:
        while True:
            with self.lock:
                if not self.running or self.cap is None:
                    break
                cap = self.cap

            success, frame = cap.read()
            if not success:
                if self.is_video_file:
                    # End of video — loop back to the first frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.latest_frame = make_placeholder_frame("Unable to read from webcam.")
                    time.sleep(0.2)
                    continue

            try:
                results = self.model.predict(frame, conf=CONFIDENCE_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
                result = results[0]
                annotated = result.plot()

                detections = []
                if result.boxes is not None and len(result.boxes) > 0:
                    names = result.names or {}
                    for box in result.boxes:
                        cls_id    = int(box.cls[0])
                        confidence = float(box.conf[0])
                        raw_label  = names.get(cls_id, str(cls_id))
                        label      = normalize_label(raw_label)
                        detections.append({"label": label, "confidence": confidence})

                now = time.time()
                if detections and (now - self.last_event_at) >= EVENT_THROTTLE_S:
                    top_det = max(detections, key=lambda item: item["confidence"])
                    self._add_event(top_det["label"], top_det["confidence"])
                    self._update_stats_from_boxes(detections)
                    self.last_event_at = now

                ok, buffer = cv2.imencode(".jpg", annotated)
                if ok:
                    self.latest_frame = buffer.tobytes()

            except Exception as exc:
                self.latest_frame = make_placeholder_frame(f"Inference error: {str(exc)[:60]}")
                log.error("Inference error: %s", exc)
                time.sleep(0.2)

    # ── Stats payload ─────────────────────────────────────────────────────────

    def get_stats_payload(self) -> Dict:
        total    = self.stats["total"]
        helmet   = self.stats["helmet"]
        no_helmet = self.stats["no_helmet"]
        compliance = int(round((helmet / total) * 100)) if total else 0
        return {
            "running":        self.running,
            "model_loaded":   self.model_loaded,
            "model_error":    self.model_error,
            "total":          total,
            "helmet":         helmet,
            "no_helmet":      no_helmet,
            "other":          self.stats["other"],
            "compliance":     compliance,
            "recent_events":  list(self.recent_events),
            "is_video_file":  self.is_video_file,
            "video_filename": self.video_filename,
        }

    # ── Image inference ───────────────────────────────────────────────────────

    def detect_image_bytes(self, image_bytes: bytes, filename: str = "") -> Dict:
        # File-type guard
        if filename and not _allowed_image(filename):
            return {"ok": False, "message": f"Unsupported file type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"}

        if not self.model_loaded:
            self.load_model()
        if not self.model_loaded:
            return {"ok": False, "message": self.model_error or "Model could not be loaded."}

        np_arr      = np.frombuffer(image_bytes, dtype=np.uint8)
        image_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_array is None:
            return {"ok": False, "message": "Invalid or corrupted image file."}

        results  = self.model.predict(image_array, conf=CONFIDENCE_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        result   = results[0]
        annotated = result.plot()

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names or {}
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                confidence = float(box.conf[0])
                raw_label  = names.get(cls_id, str(cls_id))
                label      = normalize_label(raw_label)
                detections.append({"label": label, "confidence": int(round(confidence * 100))})

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            return {"ok": False, "message": "Could not encode detection result."}

        encoded_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
        log.info("Image inference completed — %d detection(s)", len(detections))
        return {
            "ok":           True,
            "message":      "Image detection completed successfully.",
            "image_base64": encoded_image,
            "detections":   detections,
        }


backend = HelmetDetectionBackend()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def root() -> Response:
    return send_from_directory(APP_DIR, "index.html")


@app.route("/start_camera", methods=["POST"])
def start_camera() -> Response:
    return jsonify(backend.start())  # webcam


@app.route("/upload_video", methods=["POST"])
def upload_video() -> Response:
    """
    Accept a video file, save it to uploaded_videos/, then start detection on it.
    The video loops automatically when it reaches the end.
    """
    uploaded_file = request.files.get("video")
    if uploaded_file is None:
        return jsonify({"ok": False, "message": "No video file received."}), 400

    filename = uploaded_file.filename or ""
    if not _allowed_video(filename):
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        return jsonify({"ok": False, "message": f"Unsupported video type. Allowed: {allowed}"}), 400

    # Stop any running detection first
    if backend.running:
        backend.stop()

    # Save the uploaded video
    safe_name = os.path.basename(filename)          # strip any path traversal
    save_path = os.path.join(VIDEO_UPLOAD_DIR, safe_name)
    uploaded_file.save(save_path)
    log.info("Video saved to %s", save_path)

    result = backend.start(source=save_path)
    return jsonify(result)


@app.route("/stop_camera", methods=["POST"])
def stop_camera() -> Response:
    return jsonify(backend.stop())


@app.route("/reset_stats", methods=["POST"])
def reset_stats() -> Response:
    """Reset detection counters without touching the camera stream."""
    return jsonify(backend.reset_stats())


@app.route("/stats")
def stats() -> Response:
    return jsonify(backend.get_stats_payload())


@app.route("/detect_image", methods=["POST"])
def detect_image() -> Response:
    uploaded_file = request.files.get("image")
    if uploaded_file is None:
        return jsonify({"ok": False, "message": "No image file received."}), 400
    return jsonify(backend.detect_image_bytes(uploaded_file.read(), uploaded_file.filename or ""))


@app.route("/video_feed")
def video_feed() -> Response:
    frame_delay = 1.0 / STREAM_FPS

    def generate():
        while True:
            frame = backend.latest_frame
            if not frame:
                frame = make_placeholder_frame("No frame available yet.")
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(frame_delay)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(_) -> Response:
    return jsonify({"ok": False, "message": f"File too large. Maximum video size is {MAX_VIDEO_MB} MB."}), 413


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting Helmet Detection Server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=DEBUG_MODE)