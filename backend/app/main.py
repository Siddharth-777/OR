#IMPORTS
from fastapi import FastAPI
from app.config import SUPABASE_URL

app=FastAPI(title="OPTI-ROLL",version="0.1")

@app.get("/health")
def health():
    return{
        "status" : True,
        "name" : "Opti-roll backend"
    }

