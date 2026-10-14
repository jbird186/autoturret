"""Webcam face detection + tracking (MediaPipe).

Every DETECTION_INTERVAL frames, faces are detected with MediaPipe's
BlazeFace detector. New faces get a tracker, which then follows them on
every frame and reports position and (smoothed) velocity.

Install:
    pip install mediapipe opencv-contrib-python
"""

from __future__ import annotations

import itertools
import os
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Configuration ---------------------------------------------------------

MODEL_PATH = "blaze_face_short_range.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
MIN_DETECTION_CONFIDENCE = 0.5

DETECTION_SCALE = 0.75     # Downscale factor for detection frames
DETECTION_INTERVAL = 5     # Run detection every N frames
EMA_ALPHA = 0.4            # Velocity smoothing (higher = more responsive)
ARROW_SCALE = 0.25         # Velocity arrow length: px per (px/s)

COLOR_BOX = (0, 165, 255)
COLOR_CENTER = (0, 255, 255)
COLOR_VELOCITY = (255, 0, 255)

Box = tuple[int, int, int, int]  # x, y, w, h
Point = tuple[int, int]

_next_id = itertools.count(1)


# --- Data ------------------------------------------------------------------

@dataclass
class TrackedFace:
    tracker: cv2.Tracker
    box: Box
    id: int = field(default_factory=lambda: next(_next_id))
    previous_center: Point | None = None
    previous_time: float | None = None
    smoothed_velocity: tuple[float, float] = (0.0, 0.0)

    @property
    def center(self) -> Point:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2

    @property
    def speed(self) -> float:
        vx, vy = self.smoothed_velocity
        return (vx ** 2 + vy ** 2) ** 0.5


# --- Setup -----------------------------------------------------------------

def create_face_detector() -> vision.FaceDetector:
    """Create a MediaPipe face detector, downloading the model if needed."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face detection model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    return vision.FaceDetector.create_from_options(options)


def create_tracker() -> cv2.Tracker:
    """Create a CSRT (preferred) or KCF tracker; APIs vary by OpenCV version."""
    legacy = getattr(cv2, "legacy", None)
    candidates = [
        (cv2, "TrackerCSRT_create"),
        (legacy, "TrackerCSRT_create"),
        (legacy, "TrackerKCF_create"),
        (cv2, "TrackerKCF_create"),
    ]

    for module, name in candidates:
        if module is not None and hasattr(module, name):
            return getattr(module, name)()

    raise RuntimeError(
        "No CSRT or KCF tracker available. Try: pip install opencv-contrib-python"
    )


# --- Tracking --------------------------------------------------------------

def calculate_velocity(
    previous: Point | None, current: Point, dt: float
) -> tuple[float, float]:
    """Return (vx, vy) in pixels per second."""
    if previous is None or dt <= 0:
        return 0.0, 0.0

    return (current[0] - previous[0]) / dt, (current[1] - previous[1]) / dt


def clamp_box(box: Box, width: int, height: int) -> Box | None:
    """Clip a box to the frame; return None if nothing is left."""
    x, y, w, h = (int(v) for v in box)
    x2, y2 = min(x + w, width), min(y + h, height)
    x, y = max(0, x), max(0, y)
    w, h = x2 - x, y2 - y

    return (x, y, w, h) if w > 0 and h > 0 else None


def update_face(face: TrackedFace, frame, now: float) -> bool:
    """Advance one tracker. Returns False if the track was lost."""
    ok, raw_box = face.tracker.update(frame)
    if not ok:
        return False

    box = clamp_box(raw_box, *frame.shape[1::-1])
    if box is None:
        return False

    face.box = box
    center = face.center

    if face.previous_time is None or face.previous_center is None:
        # First measurement: no velocity estimate yet
        face.smoothed_velocity = (0.0, 0.0)
    else:
        raw_vx, raw_vy = calculate_velocity(
            face.previous_center, center, now - face.previous_time
        )
        svx, svy = face.smoothed_velocity
        face.smoothed_velocity = (
            EMA_ALPHA * raw_vx + (1 - EMA_ALPHA) * svx,
            EMA_ALPHA * raw_vy + (1 - EMA_ALPHA) * svy,
        )

    face.previous_center = center
    face.previous_time = now
    return True


def update_trackers(faces: list[TrackedFace], frame) -> list[TrackedFace]:
    """Update all trackers and drop any that were lost."""
    now = time.perf_counter()
    return [face for face in faces if update_face(face, frame, now)]


def is_already_tracked(box: Box, faces: list[TrackedFace]) -> bool:
    """True if a tracked face's center is near this box's center."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2

    return any(
        abs(cx - (f.box[0] + f.box[2] / 2)) < w * 0.5
        and abs(cy - (f.box[1] + f.box[3] / 2)) < h * 0.5
        for f in faces
    )


def start_tracking(faces: list[TrackedFace], frame, box: Box) -> None:
    """Track a detected face unless it's already tracked."""
    if box[2] <= 0 or box[3] <= 0 or is_already_tracked(box, faces):
        return

    tracker = create_tracker()
    if tracker.init(frame, box) is not False:
        faces.append(TrackedFace(tracker, box))


# --- Detection -------------------------------------------------------------

def detect_faces(frame, detector: vision.FaceDetector) -> list[Box]:
    """Detect faces; boxes are returned at full resolution."""
    small = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    boxes = []
    for det in result.detections:
        bb = det.bounding_box
        box = clamp_box((bb.origin_x, bb.origin_y, bb.width, bb.height), width, height)
        if box is not None:
            boxes.append(tuple(int(v / DETECTION_SCALE) for v in box))
    return boxes


# --- Drawing / output ------------------------------------------------------

def draw_tracked(frame, face: TrackedFace) -> None:
    x, y, w, h = face.box
    cx, cy = face.center
    vx, vy = face.smoothed_velocity

    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BOX, 2)
    cv2.circle(frame, face.center, 4, COLOR_CENTER, -1)

    arrow_end = (int(cx + vx * ARROW_SCALE), int(cy + vy * ARROW_SCALE))
    cv2.arrowedLine(frame, face.center, arrow_end, COLOR_VELOCITY, 2, tipLength=0.25)

    text = f"Face {face.id} X:{cx} Y:{cy} Vx:{vx:.0f} Vy:{vy:.0f}"
    cv2.putText(
        frame, text, (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 2,
    )


def log_face(face: TrackedFace) -> None:
    cx, cy = face.center
    vx, vy = face.smoothed_velocity
    print(
        f"Face {face.id} | Position=({cx}, {cy}) px | "
        f"Velocity=({vx:.1f}, {vy:.1f}) px/s | Speed={face.speed:.1f} px/s"
    )


# --- Main loop -------------------------------------------------------------

def main() -> None:
    detector = create_face_detector()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam")

    tracked: list[TrackedFace] = []
    frame_count = 0

    print("Press 'q' to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame_count += 1

            # Every frame: advance trackers, then log and draw them
            tracked = update_trackers(tracked, frame)
            for face in tracked:
                log_face(face)
                draw_tracked(frame, face)

            # Periodically: detect faces, start tracking new ones
            if frame_count % DETECTION_INTERVAL == 0:
                for box in detect_faces(frame, detector):
                    start_tracking(tracked, frame, box)

            cv2.imshow("Face Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()