import uuid
from datetime import datetime,timezone

from fastapi import APIRouter,UploadFile,File,HTTPException
from app.supabase_client import get_supabase
from app.config import SUPABASE_BUCKET

router=APIRouter()

@router.post("/presence/upload")
async def upload_video(video:UploadFile=File(...)):
    if not video.filename:
        raise HTTPException(status_code=400,detail="No filename")

    name=video.filename.lower()
    if not (name.endswith(".mp4") or name.endswith(".mov") or name.endswith(".mkv") or name.endswith(".avi")):
        raise HTTPException(status_code=400,detail="Unsupported video type")

    job_id=str(uuid.uuid4())
    ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    storage_path=f"{ts}/{job_id}_{video.filename}"

    data=await video.read()
    if not data:
        raise HTTPException(status_code=400,detail="Empty file")

    sb=get_supabase()

    # upload to storage
    sb.storage.from_(SUPABASE_BUCKET).upload(
        path=storage_path,
        file=data,
        file_options={
            "content-type":video.content_type or "application/octet-stream",
            "upsert":False
        }
    )

    # save job metadata
    sb.table("presence_jobs").insert({
        "job_id":job_id,
        "filename":video.filename,
        "bucket":SUPABASE_BUCKET,
        "storage_path":storage_path,
        "status":"uploaded"
    }).execute()

    return{
        "status":True,
        "job_id":job_id,
        "state":"uploaded"
    }

@router.get("/presence/status/{job_id}")
def status(job_id:str):
    sb=get_supabase()
    res=sb.table("presence_jobs").select("*").eq("job_id",job_id).single().execute()

    if not res.data:
        raise HTTPException(status_code=404,detail="job not found")

    return{
        "status":True,
        "job_id":job_id,
        "state":res.data["status"],
        "created_at":res.data["created_at"]
    }
