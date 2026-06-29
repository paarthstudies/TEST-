from fastapi import FastAPI
from backend.api.routes_ingest import router as ingest_router
from backend.api.routes_chat import router as chat_router

app = FastAPI(title="Offline Tabular & Document RAG Backend")

app.include_router(ingest_router)
app.include_router(chat_router)
