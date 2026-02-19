# app/services/revalidate_service.py
import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

from app.supabase_client import get_supabase
from app.services.face_service import (
    crop_face_gray,
    biggest_box,
    compute_trust,
    sb_upload,
    WEIGHTS_PATH,
    SNAPS_BUCKET,
    MODELS_BUCKET,
    MODEL_PATH,
    FACE_THRESHOLD,
    CAMERA_INDEX
)

REVALIDATE_SECONDS = 2


def micro_scan_trust():
    sb = get_supabase()

    # load LBPH
    raw = sb.storage.from_(MODELS_BUCKET).download(MODEL_PATH)
    if not raw:
        return None

    tmp="_tmp_lbph.yml"
    with open(tmp,"wb") as f:
        f.write(raw)

    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(tmp)

    detector=YOLO(str(WEIGHTS_PATH))
    cap=cv2.VideoCapture(CAMERA_INDEX)

    frames_total=0
    frames_with_face=0
    frames_recognized=0
    best_conf=0
    centers=[]
    best_face=None

    start=time.time()

    while time.time()-start < REVALIDATE_SECONDS:
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
        centers.append(((box[0]+box[2])/2,(box[1]+box[3])/2))

        face=crop_face_gray(frame,box)
        if face is None:
            continue

        label,dist=recognizer.predict(face)
        confidence=100-float(dist)

        if confidence>best_conf:
            best_conf=confidence
            best_face=face.copy()

        if dist<=FACE_THRESHOLD:
            frames_recognized+=1

    cap.release()
    cv2.destroyAllWindows()

    stability=0.5
    if len(centers)>=2:
        diffs=[np.hypot(centers[i][0]-centers[i-1][0],centers[i][1]-centers[i-1][1]) for i in range(1,len(centers))]
        jitter=float(np.mean(diffs)) if diffs else 9999
        stability=float(1/(1+(jitter/50)))

    trust,reason,_,_=compute_trust(
        frames_total,frames_with_face,frames_recognized,best_conf,stability
    )

    return trust, best_face

    
def revalidate_attendance(attendance_id:int):
    sb=get_supabase()

    row=sb.table("attendance").select("*").eq("id",attendance_id).single().execute()
    if not row.data:
        return {"ok":False,"error":"Attendance not found"}

    current=row.data
    old_status=current["status"]
    attempts=current.get("revalidation_attempts",0)

    result=micro_scan_trust()
    if result is None:
        return {"ok":False,"error":"Camera/model failure"}

    new_trust,face=result

    # ---- STATUS TRANSITION RULES ----
    if new_trust >= 85:
        new_status="present"

    elif new_trust >= 70:
        new_status="present_soft"

    else:
        # downgrade ladder
        if old_status=="present_soft":
            new_status="suspicious"
        elif old_status=="suspicious" and attempts>=1:
            new_status="absent"
        else:
            new_status="suspicious"

    # snapshot
    snapshot_path=None
    if face is not None:
        ts=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        snapshot_path=f"revalidation/{attendance_id}_{ts}.jpg"
        ok,buf=cv2.imencode(".jpg",face)
        if ok:
            sb_upload(sb,SNAPS_BUCKET,snapshot_path,buf.tobytes(),"image/jpeg")

    # update DB
    sb.table("attendance").update({
        "status":new_status,
        "revalidation_attempts":attempts+1
    }).eq("id",attendance_id).execute()

    return {
        "ok":True,
        "old_status":old_status,
        "new_status":new_status,
        "new_trust":new_trust,
        "attempt":attempts+1,
        "snapshot_path":snapshot_path
    }
