# Local Offline RAG Backend

A **100 % offline** Retrieval-Augmented Generation (RAG) system built with:

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| LLM | Ollama (llama3 / mistral) |
| Embeddings | `BAAI/bge-small-en-v1.5` via Hugging Face sentence-transformers |
| Vector DB | LanceDB (embedded) |
| Keyword Search | BM25 via Tantivy FTS index in LanceDB |

---

## Project Structure

```
rag_chatbot/
├── main.py          # FastAPI app – /upload & /chat endpoints
├── config.py        # All tuneable constants (paths, model names, alpha values)
├── ingest.py        # Dual-route document parsing & chunking
├── embedder.py      # HuggingFace sentence-transformer singleton
├── database.py      # LanceDB lifecycle, schema, ingest, hybrid search
├── llm.py           # Ollama client + RAG prompt construction
├── requirements.txt # Python dependencies
└── README.md
```

---

## Prerequisites

### 1. Python ≥ 3.10

### 2. Ollama
```bash
# Install from https://ollama.com/download
# Then pull the model:
ollama pull llama3
# Start the daemon (usually auto-started after install):
ollama serve
```

### 3. Python dependencies
```bash
pip install -r requirements.txt
```

> **First run note**: `sentence-transformers` will automatically download
> `BAAI/bge-small-en-v1.5` (~120 MB) to `~/.cache/huggingface` on the first
> import.  All subsequent runs are fully offline.

---

## Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or directly:
```bash
python main.py
```

Interactive API docs are available at **http://localhost:8000/docs**.

---

## API Reference

### `POST /upload`

Upload and ingest a document.

**Supported formats:** `.pdf`, `.txt`, `.pptx`, `.csv`, `.xlsx`

```bash
curl -X POST http://localhost:8000/upload \
     -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "message": "Document ingested successfully.",
  "filename": "document.pdf",
  "source_type": "standard",
  "chunks_ingested": 42
}
```

---

### `POST /chat`

Ask a question over the ingested corpus.

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the salary of Alice Johnson?", "source_type": "tabular"}'
```

**Request body:**
```json
{
  "query": "Your question here",
  "source_type": "standard",   // "standard" | "tabular"
  "top_k": 6                   // optional, default 6
}
```

**Response:**
```json
{
  "answer": "The salary of Alice Johnson is $95,000.",
  "source_type_used": "tabular",
  "alpha_used": 0.3,
  "chunks_retrieved": 6,
  "sources": [
    {
      "text": "Regarding Alice Johnson: The Salary for Alice Johnson is 95000.",
      "source_file": "employees.csv",
      "source_type": "tabular",
      "chunk_index": 3,
      "metadata": {}
    }
  ]
}
```

---

## Hybrid Search Architecture

LanceDB's hybrid search merges two ranked lists using a `LinearCombinationReranker`:

```
final_score = alpha × dense_score + (1 - alpha) × BM25_score
```

| `source_type` | alpha | Dense weight | BM25 weight | Rationale |
|---------------|-------|-------------|-------------|-----------|
| `standard`    | 0.6   | 60 %        | 40 %        | Prose benefits from semantic similarity |
| `tabular`     | 0.3   | 30 %        | 70 %        | Serialised rows are keyword-rich; BM25 finds exact entity/value matches |

---

## Document Parsing Strategy

### Standard (PDF / TXT / PPTX)
Text is extracted and split with a **Recursive Character Text Splitter** that
respects paragraph → sentence → word boundaries, with configurable chunk size
and overlap (see `config.py`).

### Tabular (CSV / XLSX)
Rows are converted with **Entity-Anchored Serialisation**:

```
"Regarding Alice Johnson: The Age for Alice Johnson is 34.
 The Department for Alice Johnson is Engineering.
 The Salary for Alice Johnson is 95000."
```

Repeating the entity name next to every fact maximises BM25 recall for
entity-centric queries while preserving semantic coherence for dense search.

---

## Configuration

Edit `config.py` to tune:

| Constant | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | HF model for embeddings |
| `OLLAMA_MODEL` | `llama3` | Ollama model tag |
| `HYBRID_ALPHA_STANDARD` | `0.6` | Dense weight for standard queries |
| `HYBRID_ALPHA_TABULAR` | `0.3` | Dense weight for tabular queries |
| `TOP_K` | `6` | Retrieved chunks per query |
| `CHUNK_SIZE` | `512` | Characters per chunk (standard docs) |
| `CHUNK_OVERLAP` | `64` | Overlap characters between chunks |
