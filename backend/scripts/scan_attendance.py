# scripts/scan_attendance.py
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import json
import time
from datetime import datetime

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
    FACE_THRESHOLD,
)

SNAPS_BUCKET="attendance-snaps"

def sb_download(sb,bucket,path):
    try:
        return sb.storage.from_(bucket).download(path)
    except Exception:
        return None

def sb_upload(sb,bucket,path,data_bytes,content_type="application/octet-stream"):
    storage=sb.storage.from_(bucket)
    try:
        storage.upload(
            path=path,
            file=data_bytes,
            file_options={"content-type":str(content_type)},
        )
        return
    except Exception:
        pass
    storage.update(
        path=path,
        file=data_bytes,
        file_options={"content-type":str(content_type)},
    )

def load_labels(sb):
    raw=sb_download(sb,SUPABASE_MODELS_BUCKET,FACE_LABELS_PATH)
    if not raw:
        return None
    try:
        obj=json.loads(raw.decode("utf-8"))
        if "label_to_person" not in obj:
            return None
        return obj
    except Exception:
        return None

def load_lbph(sb):
    raw=sb_download(sb,SUPABASE_MODELS_BUCKET,FACE_MODEL_PATH)
    if not raw:
        return None
    tmp="_tmp_lbph.yml"
    with open(tmp,"wb") as f:
        f.write(raw)
    model=cv2.face.LBPHFaceRecognizer_create()
    model.read(tmp)
    return model

def biggest_box_xyxy(xyxy_list):
    if not xyxy_list:
        return None
    areas=[(b[2]-b[0])*(b[3]-b[1]) for b in xyxy_list]
    return xyxy_list[int(np.argmax(areas))]

def crop_face_gray(frame,box):
    x1,y1,x2,y2=[int(v) for v in box]
    x1=max(0,x1);y1=max(0,y1);x2=min(frame.shape[1],x2);y2=min(frame.shape[0],y2)
    if x2-x1<30 or y2-y1<30:
        return None
    face=frame[y1:y2,x1:x2]
    gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(160,160))
    return gray

def main():
    sb=get_supabase()

    labels=load_labels(sb)
    if not labels:
        print("No labels found. Register someone first.")
        raise SystemExit(1)

    lbph=load_lbph(sb)
    if not lbph:
        print("No model found. Register someone first.")
        raise SystemExit(1)

    weights=Path("scripts/weights/yolov8n-face.pt")
    if not weights.exists():
        print("Missing YOLO weights: scripts/weights/yolov8n-face.pt")
        raise SystemExit(1)

    detector=YOLO(str(weights))

    cap=cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera not opening.")
        raise SystemExit(1)

    start=time.time()
    best=None  # (person_id,name,label,confidence,dist,face_gray)

    print(f"Scanning for {SCAN_SECONDS} seconds...")

    while time.time()-start<SCAN_SECONDS:
        ok,frame=cap.read()
        if not ok:
            continue

        res=detector(frame,verbose=False)[0]
        boxes=[]
        if res.boxes is not None and len(res.boxes)>0:
            boxes=res.boxes.xyxy.cpu().numpy().tolist()

        box=biggest_box_xyxy(boxes)
        if box is None:
            cv2.imshow("Scan",frame);cv2.waitKey(1)
            continue

        face_gray=crop_face_gray(frame,box)
        if face_gray is None:
            cv2.imshow("Scan",frame);cv2.waitKey(1)
            continue

        label,dist=lbph.predict(face_gray)  # smaller dist = better
        confidence=max(0.0,100.0-float(dist))

        if float(dist)<=float(FACE_THRESHOLD):
            info=labels["label_to_person"].get(str(label))
            if info:
                cand=(info["person_id"],info["name"],int(label),confidence,float(dist),face_gray.copy())
                if (best is None) or (cand[3]>best[3]):
                    best=cand

        cv2.imshow("Scan",frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not best:
        print("No registered face recognized. Exiting.")
        raise SystemExit(1)

    person_id,name,label,confidence,dist,face_gray=best

    sb.table("attendance").insert({
        "person_id":person_id,
        "name":name,
        "confidence":float(confidence),
        "status":"present"
    }).execute()

    ts=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snap_path=f"label_{label}/{ts}.jpg"

    ok,buf=cv2.imencode(".jpg",face_gray)
    if ok:
        sb_upload(sb,SNAPS_BUCKET,snap_path,buf.tobytes(),content_type="image/jpeg")
        print(f"Face snapshot saved ✅ -> {SNAPS_BUCKET}/{snap_path}")
    else:
        print("Attendance marked ✅ but snapshot encode failed.")

    print("Attendance marked ✅")
    print({"person_id":person_id,"name":name,"label":label,"confidence":confidence,"distance":dist})

if __name__=="__main__":
    main()
