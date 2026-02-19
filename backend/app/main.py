from fastapi import FastAPI
from app.routes import router
from app.services.scheduler_service import start_scheduler

app = FastAPI(title="OPTI-ROLL")

app.include_router(router)

@app.on_event("startup")
def start_background_jobs():
    start_scheduler()
