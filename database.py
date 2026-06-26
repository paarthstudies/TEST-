"""
database.py – LanceDB lifecycle management and search utilities.

Responsibilities
────────────────
1. Open / create the LanceDB database and the chunk table on startup.
2. Define the Arrow schema so LanceDB knows the vector dimension upfront.
3. Write batches of chunks (text + vector + metadata) into the table.
4. Rebuild the Full-Text Search (FTS / BM25) index after each ingestion.
5. Execute hybrid vector + BM25 searches with caller-supplied alpha weights.

LanceDB hybrid search overview
───────────────────────────────
LanceDB's hybrid search merges two ranked lists:
  • Dense (ANN) search  – cosine similarity over the `vector` column.
  • Sparse (FTS) search – BM25 over the `text` column via the Tantivy index.

The `rerank(LinearCombinationReranker(weight=alpha))` call controls blending:
  alpha = 1.0  → 100 % dense  (pure vector)
  alpha = 0.0  → 100 % BM25   (pure keyword)

We expose two named presets:
  "standard" → alpha = HYBRID_ALPHA_STANDARD (0.6)  60:40 dense:BM25
  "tabular"  → alpha = HYBRID_ALPHA_TABULAR  (0.3)  30:70 dense:BM25

Why tabular queries lean heavier on BM25:
  Entity-anchored serialised rows are keyword-rich.  A query like
  "What is Alice Johnson's salary?" will match the BM25 index very
  precisely, while dense similarity might drift toward semantically
  related but factually wrong rows.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import lancedb
import pyarrow as pa
from lancedb.rerankers import LinearCombinationReranker

from config import (
    EMBEDDING_DIM,
    HYBRID_ALPHA_STANDARD,
    HYBRID_ALPHA_TABULAR,
    LANCEDB_URI,
    TABLE_NAME,
    TOP_K,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons – initialised once via `initialise_db()`
# ---------------------------------------------------------------------------

_db: Optional[lancedb.LanceDBConnection] = None
_table: Optional[lancedb.table.Table] = None


# ---------------------------------------------------------------------------
# Arrow schema
# ---------------------------------------------------------------------------

def _build_schema() -> pa.Schema:
    """
    Define the table schema for LanceDB.

    Columns:
      id          – auto-incrementing surrogate key (string UUID).
      vector      – dense embedding (FixedSizeList of float32).
      text        – raw chunk text; also the FTS search column.
      source_file – original file name.
      source_type – "standard" or "tabular".
      chunk_index – position of this chunk within the source document.
      metadata    – JSON-encoded dict for any extra per-chunk data.
    """
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            pa.field("text", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("metadata", pa.string()),  # JSON blob
        ]
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def initialise_db() -> None:
    """
    Open the LanceDB connection and ensure the chunk table exists.

    Called once from the FastAPI `lifespan` startup handler so the table
    is ready before the first request arrives.
    """
    global _db, _table

    logger.info("Connecting to LanceDB at: %s", LANCEDB_URI)
    _db = lancedb.connect(LANCEDB_URI)

    existing_tables = _db.table_names()
    if TABLE_NAME in existing_tables:
        logger.info("Opening existing table '%s'.", TABLE_NAME)
        _table = _db.open_table(TABLE_NAME)
    else:
        logger.info("Creating new table '%s' with schema.", TABLE_NAME)
        # Create with an empty initial batch so LanceDB registers the schema
        empty_batch = pa.table(
            {
                "id": pa.array([], type=pa.string()),
                "vector": pa.array(
                    [], type=pa.list_(pa.float32(), EMBEDDING_DIM)
                ),
                "text": pa.array([], type=pa.string()),
                "source_file": pa.array([], type=pa.string()),
                "source_type": pa.array([], type=pa.string()),
                "chunk_index": pa.array([], type=pa.int32()),
                "metadata": pa.array([], type=pa.string()),
            }
        )
        _table = _db.create_table(TABLE_NAME, data=empty_batch, schema=_build_schema())
        logger.info("Table '%s' created.", TABLE_NAME)


def get_table() -> lancedb.table.Table:
    """Return the active LanceDB table, raising if not initialised."""
    if _table is None:
        raise RuntimeError(
            "LanceDB table is not initialised. "
            "Make sure `initialise_db()` was called at startup."
        )
    return _table


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def ingest_chunks(
    texts: List[str],
    embeddings: List[List[float]],
    metadata_list: List[Dict[str, Any]],
) -> int:
    """
    Write a batch of chunks into the LanceDB table and rebuild the FTS index.

    Args:
        texts:         Raw chunk strings.
        embeddings:    Dense float vectors, one per chunk.
        metadata_list: Metadata dicts, one per chunk.

    Returns:
        Number of chunks successfully written.
    """
    import uuid

    table = get_table()

    if not texts:
        logger.warning("ingest_chunks called with empty texts list – skipping.")
        return 0

    rows: List[Dict[str, Any]] = []
    for text, vector, meta in zip(texts, embeddings, metadata_list):
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "vector": vector,
                "text": text,
                "source_file": meta.get("source_file", "unknown"),
                "source_type": meta.get("source_type", "standard"),
                "chunk_index": int(meta.get("chunk_index", 0)),
                # Serialise the full metadata dict as JSON for later inspection
                "metadata": json.dumps(meta),
            }
        )

    table.add(rows)
    logger.info("Ingested %d chunks into '%s'.", len(rows), TABLE_NAME)

    # -------------------------------------------------------------------
    # Rebuild Full-Text Search (BM25) index
    # -------------------------------------------------------------------
    # `create_fts_index` builds or replaces the Tantivy inverted index
    # on the `text` column.  Setting `replace=True` makes this call
    # idempotent – safe to call on every upload without manual cleanup.
    # -------------------------------------------------------------------
    table.create_fts_index("text", replace=True)
    logger.info("FTS index rebuilt on column 'text'.")

    return len(rows)


# ---------------------------------------------------------------------------
# Read / search path
# ---------------------------------------------------------------------------

def hybrid_search(
    query_text: str,
    query_vector: List[float],
    source_type: str,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Perform a hybrid (dense + BM25) search and return the top-K results.

    Alpha selection
    ───────────────
    The `LinearCombinationReranker` merges normalised dense and BM25 scores:
      final_score = alpha * dense_score + (1 - alpha) * bm25_score

    We choose alpha based on the expected query/document type:
      "standard" → alpha = 0.6  (60 % dense, 40 % BM25)
      "tabular"  → alpha = 0.3  (30 % dense, 70 % BM25)

    Args:
        query_text:   Raw query string for BM25 matching.
        query_vector: Dense embedding of the query for ANN search.
        source_type:  "standard" or "tabular" – drives alpha selection.
        top_k:        Maximum number of results to return.

    Returns:
        List of result dicts with keys: text, source_file, source_type,
        chunk_index, metadata, _distance, _score.
    """
    table = get_table()

    # ------------------------------------------------------------------
    # Pick the hybrid alpha based on source_type
    # ------------------------------------------------------------------
    if source_type == "tabular":
        alpha = HYBRID_ALPHA_TABULAR      # 0.3 → heavy BM25 (70 %)
    else:
        alpha = HYBRID_ALPHA_STANDARD     # 0.6 → balanced dense-leaning

    logger.info(
        "Hybrid search | source_type=%s | alpha=%.1f | top_k=%d",
        source_type,
        alpha,
        top_k,
    )

    # ------------------------------------------------------------------
    # LanceDB hybrid search pipeline
    # ------------------------------------------------------------------
    # 1. `.search([query_vector], query_type="hybrid")` kicks off both
    #    the ANN vector pass and the full-text BM25 pass simultaneously.
    # 2. `.rerank(LinearCombinationReranker(weight=alpha))` blends scores.
    # 3. `.limit(top_k)` caps the result set.
    # 4. `.to_list()` materialises the result as Python dicts.
    # ------------------------------------------------------------------
    reranker = LinearCombinationReranker(weight=alpha)

    results = (
        table.search(query_text, query_type="hybrid")
        .rerank(reranker)
        .limit(top_k)
        .to_list()
    )

    logger.info("Hybrid search returned %d results.", len(results))
    return results
