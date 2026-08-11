from fastapi import FastAPI

from app.api.routes import health

app = FastAPI(title="NiftyBridge RAG Chatbot")

app.include_router(health.router, prefix="/api")