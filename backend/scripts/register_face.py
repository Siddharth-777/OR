# scripts/register_face.py
import os
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from app.supabase_client import get_supabase
from app.config import (
    SUPABASE_MODELS_BUCKET,
    FACE_MODEL_PATH,
    FACE_LABELS_PATH,
    CAMERA_INDEX,
    SCAN_SECONDS,
)

# -----------------------------
# SUPABASE STORAGE HELPERS
# -----------------------------
def sb_download(sb, bucket, path):
    try:
        return sb.storage.from_(bucket).download(path)
    except Exception:
        return None


def sb_upload(sb, bucket, path, data_bytes, content_type="application/octet-stream"):
    storage = sb.storage.from_(bucket)

    # 1) try normal upload
    try:
        storage.upload(
            path=path,
            file=data_bytes,
            file_options={"content-type": str(content_type)},
        )
        return
    except Exception:
        pass

    # 2) if already exists, update/overwrite
    try:
        storage.update(
            path=path,
            file=data_bytes,
            file_options={"content-type": str(content_type)},
        )
        return
    except Exception as e:
        raise e



# -----------------------------
# LABELS JSON (label -> person)
# -----------------------------
def load_labels(sb):
    raw = sb_download(sb, SUPABASE_MODELS_BUCKET, FACE_LABELS_PATH)
    if not raw:
        return {"label_to_person": {}}
    try:
        obj = json.loads(raw.decode("utf-8"))
        if "label_to_person" not in obj:
            obj["label_to_person"] = {}
        return obj
    except Exception:
        return {"label_to_person": {}}


def save_labels(sb, labels):
    sb_upload(
        sb,
        SUPABASE_MODELS_BUCKET,
        FACE_LABELS_PATH,
        json.dumps(labels).encode("utf-8"),
        content_type="application/json",
    )


# -----------------------------
# LBPH MODEL LOAD/SAVE
# -----------------------------
def load_lbph(sb):
    raw = sb_download(sb, SUPABASE_MODELS_BUCKET, FACE_MODEL_PATH)
    model = cv2.face.LBPHFaceRecognizer_create()
    if not raw:
        return model, False

    tmp = "_tmp_lbph.yml"
    with open(tmp, "wb") as f:
        f.write(raw)
    model.read(tmp)
    try:
        os.remove(tmp)
    except Exception:
        pass

    return model, True


def save_lbph(sb, model):
    tmp = "_tmp_lbph_out.yml"
    model.write(tmp)
    with open(tmp, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp)
    except Exception:
        pass

    sb_upload(sb, SUPABASE_MODELS_BUCKET, FACE_MODEL_PATH, data, content_type="text/yaml")


# -----------------------------
# FACE CROP HELPERS
# -----------------------------
def biggest_box_xyxy(xyxy_list):
    if not xyxy_list:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy_list]
    return xyxy_list[int(np.argmax(areas))]


def crop_face(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 - x1 < 30 or y2 - y1 < 30:
        return None

    face = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 160))
    return gray


# -----------------------------
# MAIN
# -----------------------------
def main():
    name = input("Enter person name to register: ").strip()
    if not name:
        print("No name. Exiting.")
        raise SystemExit(1)

    sb = get_supabase()

    # ✅ label allocation from DB (no duplicate)
    row = sb.table("face_registry").select("label").order("label", desc=True).limit(1).execute()
    last = row.data[0]["label"] if row.data else -1
    label = int(last) + 1

    # Insert registry row first
    reg = sb.table("face_registry").insert({"name": name, "label": label}).execute()
    person_id = reg.data[0]["person_id"]

    # Load labels.json (safe)
    labels = load_labels(sb)

    # Update mapping
    labels["label_to_person"][str(label)] = {"person_id": person_id, "name": name}

    # YOLO face weights
    weights = Path("scripts/weights/yolov8n-face.pt")
    if not weights.exists():
        print("Missing YOLO weights:")
        print("Put yolov8n-face.pt in scripts/weights/")
        # cleanup DB row
        sb.table("face_registry").delete().eq("person_id", person_id).execute()
        raise SystemExit(1)

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera not opening.")
        sb.table("face_registry").delete().eq("person_id", person_id).execute()
        raise SystemExit(1)

    start = time.time()
    faces = []

    print(f"Capturing faces for {SCAN_SECONDS} seconds... Look at camera.")

    while time.time() - start < SCAN_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue

        res = model(frame, verbose=False)[0]
        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy().tolist()

        box = biggest_box_xyxy(boxes)
        if box is None:
            cv2.imshow("Register", frame)
            cv2.waitKey(1)
            continue

        gray = crop_face(frame, box)
        if gray is None:
            cv2.imshow("Register", frame)
            cv2.waitKey(1)
            continue

        faces.append(gray)

        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Need enough samples
    if len(faces) < 10:
        print("Not enough face samples captured. Registration failed.")
        sb.table("face_registry").delete().eq("person_id", person_id).execute()
        raise SystemExit(1)

    # Train/update LBPH
    lbph, has_old = load_lbph(sb)
    y = np.array([label] * len(faces), dtype=np.int32)

    if has_old:
        lbph.update(faces, y)
    else:
        lbph.train(faces, y)

    save_lbph(sb, lbph)
    save_labels(sb, labels)

    print("Registered successfully ✅")
    print({"person_id": person_id, "name": name, "label": label})


if __name__ == "__main__":
    main()
