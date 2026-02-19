# app/services/face_service.py
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from app.supabase_client import get_supabase
from app.config import (
    FACE_THRESHOLD,
    CAMERA_INDEX,
    SCAN_SECONDS,
)

SNAPS_BUCKET = "attendance-snaps"
WEIGHTS_PATH = Path("scripts/weights/yolov8n-face.pt")


# ---------------- storage helpers ----------------
def sb_upload(sb, bucket, path, data, ctype):
    storage = sb.storage.from_(bucket)
    try:
        storage.upload(path=path, file=data, file_options={"content-type": ctype})
    except Exception:
        storage.update(path=path, file=data, file_options={"content-type": ctype})


# ---------------- vision helpers ----------------
def biggest_box(boxes):
    if not boxes:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return boxes[int(np.argmax(areas))]


def crop_face_gray(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 - x1 < 30 or y2 - y1 < 30:
        return None

    face = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 160))


# ---------------- MAIN SERVICE ----------------
def scan_and_mark_attendance():
    sb = get_supabase()

    # Load LBPH model from Supabase
    raw = sb.storage.from_("models").download("face_models/lbph_model.yml")
    if not raw:
        return {"ok": False, "error": "LBPH model missing"}

    tmp = "_tmp_lbph.yml"
    with open(tmp, "wb") as f:
        f.write(raw)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(tmp)

    if not WEIGHTS_PATH.exists():
        return {"ok": False, "error": "YOLO weights missing"}

    detector = YOLO(str(WEIGHTS_PATH))
    cap = cv2.VideoCapture(CAMERA_INDEX)

    start = time.time()
    best = None  # (label, confidence, face_gray)

    while time.time() - start < SCAN_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue

        res = detector(frame, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes else []
        box = biggest_box(boxes)
        if not box:
            continue

        face_gray = crop_face_gray(frame, box)
        if face_gray is None:
            continue

        label, dist = recognizer.predict(face_gray)
        confidence = 100.0 - float(dist)

        if dist <= FACE_THRESHOLD:
            if (best is None) or (confidence > best[1]):
                best = (int(label), confidence, face_gray.copy())

    cap.release()
    cv2.destroyAllWindows()

    if not best:
        return {"ok": False, "error": "No face recognized"}

    label, confidence, face_gray = best

    # 🔥 DB is source of truth (NO FK CRASH)
    row = (
        sb.table("face_registry")
        .select("person_id,name")
        .eq("label", label)
        .single()
        .execute()
    )

    if not row.data:
        return {"ok": False, "error": "Label not found in face_registry"}

    person_id = row.data["person_id"]
    name = row.data["name"]

    # Insert attendance
    sb.table("attendance").insert({
        "person_id": person_id,
        "name": name,
        "confidence": confidence,
        "status": "present"
    }).execute()

    # Save snapshot
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snap_path = f"label_{label}/{ts}.jpg"
    ok, buf = cv2.imencode(".jpg", face_gray)

    if ok:
        sb_upload(sb, SNAPS_BUCKET, snap_path, buf.tobytes(), "image/jpeg")

    return {
        "ok": True,
        "name": name,
        "label": label,
        "confidence": confidence,
        "snapshot_path": snap_path
    }
