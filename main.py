"""
main.py – FastAPI application entry point.

Endpoints
─────────
  POST /upload   – Ingest a document (PDF / TXT / PPTX / CSV / XLSX).
  POST /chat     – Hybrid RAG query against the ingested corpus.
  GET  /health   – Liveness probe.
  GET  /stats    – Quick stats about the current LanceDB table.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local modules
import database
import embedder
import ingest as ingest_module
import llm as llm_module
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    OLLAMA_MODEL,
    TABLE_NAME,
    TOP_K,
    UPLOAD_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed file extensions
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".ppt", ".pptx", ".csv", ".xls", ".xlsx"}


# ---------------------------------------------------------------------------
# FastAPI lifespan – startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise LanceDB and pre-load the embedding model.
    Shutdown: nothing special required (LanceDB is embedded / file-backed).
    """
    logger.info("=== RAG Backend starting up ===")
    database.initialise_db()

    # Eagerly load the embedding model so the first request is not slow
    logger.info("Pre-loading embedding model …")
    embedder.embed_texts(["warmup"])
    logger.info("=== Startup complete ===")

    yield  # Application runs here

    logger.info("=== RAG Backend shutting down ===")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Local RAG Backend",
    description=(
        "100 % offline Retrieval-Augmented Generation system.\n\n"
        "• LLM: Ollama (local)\n"
        "• Embeddings: BAAI/bge-small-en-v1.5 (Hugging Face, local)\n"
        "• Vector DB: LanceDB (embedded) with hybrid Dense + BM25 search"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins during local development – tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Pydantic models
# ===========================================================================

class UploadResponse(BaseModel):
    """Response payload for a successful document upload."""
    message: str
    filename: str
    source_type: Literal["standard", "tabular"]
    chunks_ingested: int


class ChatRequest(BaseModel):
    """Incoming payload for the /chat endpoint."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="The user's natural-language question.",
        examples=["What is Alice Johnson's salary?"],
    )
    source_type: Literal["standard", "tabular"] = Field(
        default="standard",
        description=(
            "Controls the hybrid search blend ratio.\n"
            "• 'standard' → 60 % dense / 40 % BM25 (alpha=0.6)\n"
            "• 'tabular'  → 30 % dense / 70 % BM25 (alpha=0.3)"
        ),
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description=f"Override the default top-K ({TOP_K}) for this request.",
    )


class SourceChunk(BaseModel):
    """A single retrieved context chunk returned with the chat answer."""
    text: str
    source_file: str
    source_type: str
    chunk_index: int
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    """Response payload for a /chat request."""
    answer: str
    source_type_used: str
    alpha_used: float
    chunks_retrieved: int
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    table_name: str


class StatsResponse(BaseModel):
    table_name: str
    total_chunks: int
    unique_files: List[str]


# ===========================================================================
# Helper utilities
# ===========================================================================

def _validate_extension(filename: str) -> str:
    """Return the lower-case extension or raise 400 if unsupported."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File type '{suffix}' is not supported. "
                f"Allowed types: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )
    return suffix


def _determine_source_type(suffix: str) -> Literal["standard", "tabular"]:
    """Map file extension to source_type label."""
    TABULAR = {".csv", ".xls", ".xlsx"}
    return "tabular" if suffix in TABULAR else "standard"


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe – returns 200 if the service is up."""
    return HealthResponse(
        status="ok",
        llm_model=OLLAMA_MODEL,
        table_name=TABLE_NAME,
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Return a quick summary of what is currently stored in LanceDB."""
    try:
        table = database.get_table()
        df = table.to_pandas()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not read LanceDB table: {exc}",
        )

    unique_files: List[str] = (
        df["source_file"].dropna().unique().tolist() if not df.empty else []
    )
    return StatsResponse(
        table_name=TABLE_NAME,
        total_chunks=len(df),
        unique_files=unique_files,
    )


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["RAG"],
    summary="Upload and ingest a document",
    description=(
        "Accepts a PDF, TXT, PPTX, CSV, or XLSX file. "
        "Routes to the appropriate parser, generates embeddings, "
        "and stores the chunks in LanceDB with an updated FTS index."
    ),
)
async def upload_document(file: UploadFile = File(...)):
    """
    Ingestion pipeline:

    1. Validate file extension.
    2. Save the upload to a temp file (avoids holding it entirely in memory).
    3. Route to the correct parser (standard vs. tabular).
    4. Generate dense embeddings for all chunks via the local HF model.
    5. Write chunks + vectors + metadata to LanceDB.
    6. Rebuild the BM25 / FTS index.
    7. Return a success summary.
    """
    # ── 1. Validate ──────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    suffix = _validate_extension(file.filename)
    source_type = _determine_source_type(suffix)

    logger.info("Upload received: '%s' (type=%s)", file.filename, source_type)

    # ── 2. Persist to temp file ───────────────────────────────────────────
    # We write to a NamedTemporaryFile so parsers that require a real path
    # (pypdf, pptx, pandas) can open it directly.
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=UPLOAD_DIR,
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        logger.info("Saved upload to temp file: %s", tmp_path)

        # ── 3. Parse & chunk ─────────────────────────────────────────────
        try:
            chunk_pairs = ingest_module.route_and_parse(tmp_path, CHUNK_SIZE, CHUNK_OVERLAP)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )

        if not chunk_pairs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No text could be extracted from '{file.filename}'.",
            )

        texts = [pair[0] for pair in chunk_pairs]
        metadata_list = [pair[1] for pair in chunk_pairs]

        # Override source_file in metadata to use the original upload name
        for meta in metadata_list:
            meta["source_file"] = file.filename

        # ── 4. Embed ──────────────────────────────────────────────────────
        logger.info("Generating embeddings for %d chunks …", len(texts))
        try:
            embeddings = embedder.embed_texts(texts)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding generation failed: {exc}",
            )

        # ── 5 & 6. Ingest + rebuild FTS index ────────────────────────────
        try:
            n_written = database.ingest_chunks(texts, embeddings, metadata_list)
        except Exception as exc:
            logger.error("LanceDB ingestion failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database ingestion failed: {exc}",
            )

    finally:
        # Always clean up the temp file
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
            logger.info("Temp file removed: %s", tmp_path)

    # ── 7. Respond ────────────────────────────────────────────────────────
    return UploadResponse(
        message="Document ingested successfully.",
        filename=file.filename,
        source_type=source_type,
        chunks_ingested=n_written,
    )


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["RAG"],
    summary="Ask a question over ingested documents",
    description=(
        "Performs hybrid dense+BM25 search with source_type-driven alpha blending, "
        "injects the top-K chunks into a strict RAG prompt, "
        "and returns the Ollama-generated answer."
    ),
)
async def chat(request: ChatRequest):
    """
    RAG query pipeline:

    1. Validate the request.
    2. Embed the query with the same local HF model used at ingestion.
    3. Hybrid search: dense ANN + BM25 with alpha chosen by source_type.
    4. Build a strict RAG prompt with retrieved chunks as context.
    5. Call Ollama for generation.
    6. Return the answer and source attribution.
    """
    query = request.query.strip()
    source_type = request.source_type
    k = request.top_k or TOP_K

    logger.info(
        "Chat request | source_type=%s | top_k=%d | query='%s'",
        source_type,
        k,
        query[:80],
    )

    # ── 1. Sanity: make sure there is data to search ───────────────────
    try:
        table = database.get_table()
        row_count = table.count_rows()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot access the vector store: {exc}",
        )

    if row_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The knowledge base is empty. Upload at least one document first.",
        )

    # ── 2. Embed the query ────────────────────────────────────────────
    try:
        query_vector = embedder.embed_query(query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to embed the query: {exc}",
        )

    # ── 3. Hybrid search ──────────────────────────────────────────────
    try:
        raw_results = database.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            source_type=source_type,
            top_k=k,
        )
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        )

    if not raw_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant chunks found for the given query.",
        )

    # ── 4. Build context & prompt ────────────────────────────────────
    context_texts = [r["text"] for r in raw_results]

    # ── 5. Generate answer via Ollama ────────────────────────────────
    try:
        answer = llm_module.generate_answer(question=query, context_chunks=context_texts)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    # ── 6. Assemble source attribution ───────────────────────────────
    from config import HYBRID_ALPHA_STANDARD, HYBRID_ALPHA_TABULAR
    alpha_used = HYBRID_ALPHA_TABULAR if source_type == "tabular" else HYBRID_ALPHA_STANDARD

    sources = []
    for r in raw_results:
        try:
            meta_dict = json.loads(r.get("metadata", "{}"))
        except (json.JSONDecodeError, TypeError):
            meta_dict = {}
        sources.append(
            SourceChunk(
                text=r.get("text", ""),
                source_file=r.get("source_file", "unknown"),
                source_type=r.get("source_type", "unknown"),
                chunk_index=int(r.get("chunk_index", 0)),
                metadata=meta_dict,
            )
        )

    return ChatResponse(
        answer=answer,
        source_type_used=source_type,
        alpha_used=alpha_used,
        chunks_retrieved=len(sources),
        sources=sources,
    )


# ===========================================================================
# Dev-mode entry point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
