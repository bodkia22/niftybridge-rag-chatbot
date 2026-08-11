from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, health

app = FastAPI(title="NiftyBridge RAG Chatbot")

app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

app.mount("/", StaticFiles(directory="static", html=True), name="static")