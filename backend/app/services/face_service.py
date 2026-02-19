# app/services/face_service.py
import time
from pathlib import Path
from datetime import datetime
import math

import cv2
import numpy as np
from ultralytics import YOLO

from app.supabase_client import get_supabase
from app.config import (
    FACE_THRESHOLD,
    CAMERA_INDEX,
    SCAN_SECONDS,
)

SNAPS_BUCKET="attendance-snaps"
WEIGHTS_PATH=Path("scripts/weights/yolov8n-face.pt")

MODELS_BUCKET="models"
MODEL_PATH="face_models/lbph_model.yml"


def sb_upload(sb,bucket,path,data,ctype):
    storage=sb.storage.from_(bucket)
    try:
        storage.upload(path=path,file=data,file_options={"content-type":ctype})
    except Exception:
        storage.update(path=path,file=data,file_options={"content-type":ctype})


def biggest_box(boxes):
    if not boxes:
        return None
    areas=[(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    return boxes[int(np.argmax(areas))]


def crop_face_gray(frame,box):
    x1,y1,x2,y2=[int(v) for v in box]
    x1=max(0,x1);y1=max(0,y1);x2=min(frame.shape[1],x2);y2=min(frame.shape[0],y2)
    if x2-x1<30 or y2-y1<30:
        return None
    face=frame[y1:y2, x1:x2]
    gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray,(160,160))


def clamp(x,a,b):
    return a if x<a else (b if x>b else x)


def compute_trust(frames_total,frames_with_face,frames_recognized,best_confidence,stability):
    # ratios
    face_ratio = (frames_with_face/frames_total) if frames_total>0 else 0.0
    recog_ratio = (frames_recognized/frames_with_face) if frames_with_face>0 else 0.0

    # normalize confidence: assume 0..100 already (from 100-dist)
    conf_norm = clamp(best_confidence/100.0, 0.0, 1.0)

    # stability expected 0..1 (we compute 1/(1+jitter))
    stab_norm = clamp(stability, 0.0, 1.0)

    # weighted sum (purely technical)
    score = (
        0.35*conf_norm +
        0.25*recog_ratio +
        0.20*face_ratio +
        0.20*stab_norm
    ) * 100.0

    score = round(clamp(score,0.0,100.0),2)

    # reason string
    reasons=[]
    if conf_norm>=0.85: reasons.append("high match confidence")
    elif conf_norm>=0.70: reasons.append("moderate confidence")
    else: reasons.append("low confidence")

    if recog_ratio>=0.75: reasons.append("consistent recognition")
    elif recog_ratio>=0.45: reasons.append("intermittent recognition")
    else: reasons.append("rare recognition")

    if stab_norm>=0.75: reasons.append("stable face position")
    elif stab_norm>=0.45: reasons.append("some movement")
    else: reasons.append("high movement/jitter")

    reason=", ".join(reasons)
    return score, reason, face_ratio, recog_ratio


def scan_and_mark_attendance():
    sb=get_supabase()

    # Load LBPH model
    raw=sb.storage.from_(MODELS_BUCKET).download(MODEL_PATH)
    if not raw:
        return {"ok":False,"error":"LBPH model missing (register first)"}

    tmp="_tmp_lbph.yml"
    with open(tmp,"wb") as f:
        f.write(raw)
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(tmp)

    if not WEIGHTS_PATH.exists():
        return {"ok":False,"error":"YOLO weights missing: scripts/weights/yolov8n-face.pt"}

    detector=YOLO(str(WEIGHTS_PATH))
    cap=cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return {"ok":False,"error":"Camera not opening"}

    # stats
    frames_total=0
    frames_with_face=0
    frames_recognized=0
    best=None  # (label, confidence, face_gray)
    best_confidence=0.0
    centers=[]

    start=time.time()

    while time.time()-start<SCAN_SECONDS:
        ok,frame=cap.read()
        if not ok:
            continue
        frames_total+=1

        res=detector(frame,verbose=False)[0]
        boxes=res.boxes.xyxy.cpu().numpy().tolist() if res.boxes else []
        box=biggest_box(boxes)
        if not box:
            continue

        frames_with_face+=1

        cx=(box[0]+box[2])/2.0
        cy=(box[1]+box[3])/2.0
        centers.append((cx,cy))

        face_gray=crop_face_gray(frame,box)
        if face_gray is None:
            continue

        label,dist=recognizer.predict(face_gray)
        confidence=100.0-float(dist)

        if confidence>best_confidence:
            best_confidence=confidence

        if dist<=FACE_THRESHOLD:
            frames_recognized+=1
            if (best is None) or (confidence>best[1]):
                best=(int(label),confidence,face_gray.copy())

    cap.release()
    cv2.destroyAllWindows()

    # stability (0..1)
    stability=0.0
    if len(centers)>=2:
        diffs=[math.hypot(centers[i][0]-centers[i-1][0],centers[i][1]-centers[i-1][1]) for i in range(1,len(centers))]
        jitter=float(np.mean(diffs)) if diffs else 9999.0
        stability=float(1.0/(1.0+(jitter/50.0)))

    trust_score,reason,face_ratio,recog_ratio=compute_trust(
        frames_total,frames_with_face,frames_recognized,best_confidence,stability
    )

    # ---------- POLICY LADDER ----------
    # Defaults (override with env if you added to config)
    try:
        from app.config import TRUST_PRESENT, TRUST_PRESENT_SOFT, TRUST_SUSPICIOUS
        tp=float(TRUST_PRESENT);tps=float(TRUST_PRESENT_SOFT);ts=float(TRUST_SUSPICIOUS)
    except Exception:
        tp=85.0;tps=70.0;ts=50.0

    def decide_status(t):
        if t>=tp:
            return "present"
        if t>=tps:
            return "present_soft"
        if t>=ts:
            return "suspicious"
        return "absent"

    status=decide_status(trust_score)

    # If we couldn't recognize any label, treat as absent (but still return telemetry)
    if best is None:
        return {
            "ok":False,
            "error":"No face recognized",
            "status":status,
            "trust_score":trust_score,
            "reason":reason,
            "frames_total":frames_total,
            "frames_with_face":frames_with_face,
            "frames_recognized":frames_recognized,
            "best_confidence":round(best_confidence,2),
            "stability":round(stability,3),
        }

    label,confidence,face_gray=best

    # DB source of truth
    row=sb.table("face_registry").select("person_id,name").eq("label",label).single().execute()
    if not row.data:
        return {"ok":False,"error":"Label recognized but not found in face_registry"}

    person_id=row.data["person_id"]
    name=row.data["name"]

    # snapshot
    ts_now=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_path=f"label_{label}/{ts_now}.jpg"
    ok_jpg,buf=cv2.imencode(".jpg",face_gray)
    if ok_jpg:
        sb_upload(sb,SNAPS_BUCKET,snapshot_path,buf.tobytes(),"image/jpeg")

    # Save attendance ONLY if not absent
    # (optional rule; if you want to store all attempts too, tell me)
    if status!="absent":
        payload={
            "person_id":person_id,
            "name":name,
            "confidence":float(round(confidence,2)),
            "status":status,
            "trust_score":float(trust_score),
            "frames_total":int(frames_total),
            "frames_with_face":int(frames_with_face),
            "frames_recognized":int(frames_recognized),
            "best_confidence":float(round(best_confidence,2)),
            "stability":float(round(stability,3)),
            "reason":reason,
            "snapshot_path":snapshot_path,
        }

        try:
            sb.table("attendance").insert(payload).execute()
        except Exception:
            sb.table("attendance").insert({
                "person_id":person_id,
                "name":name,
                "confidence":float(round(confidence,2)),
                "status":status,
            }).execute()

    return {
        "ok":True,
        "name":name,
        "label":label,
        "confidence":round(confidence,2),
        "trust_score":trust_score,
        "status":status,
        "reason":reason,
        "frames_total":frames_total,
        "frames_with_face":frames_with_face,
        "frames_recognized":frames_recognized,
        "best_confidence":round(best_confidence,2),
        "stability":round(stability,3),
        "snapshot_path":snapshot_path
    }
