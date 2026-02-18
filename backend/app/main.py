#IMPORTS
from fastapi import FastAPI
from app.routes import router

#APP
app=FastAPI(title="OPTI-ROLL",version="0.1")

#ROUTERS
app.include_router(router)
