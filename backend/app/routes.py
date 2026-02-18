#IMPORTS
from fastapi import APIRouter
from app.supabase_client import get_supabase
from app.config import load_env_file
from app.presence import router as presence_router

#ROUTER
router = APIRouter()
router.include_router(presence_router)


#SUPABASE-PING
@router.get("/supabase/ping")
def supabase_ping():
    sb=get_supabase()
    return{
        "status" : True,
        "Connection" : True
    }

#HEALTH
@router.get("/health")
def health():
    return{
        "status" : True,
        "name" : "Opti-roll backend"
    }