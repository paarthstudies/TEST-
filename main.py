import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
from pypdf import PdfReader
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker
import requests

app = FastAPI(title="Offline Tabular & Document RAG Backend")

# --- Configurations ---
LANCE_DB_PATH = "./.lancedb"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # Change to "mistral" or your local model if needed
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Embeddings Setup ---
registry = get_registry().get("sentence-transformers")
embedding_function = registry.create(name=EMBEDDING_MODEL_NAME, device="cpu")

# --- Database Schema ---
class DocumentChunk(LanceModel):
    vector: Vector(embedding_function.ndims()) = embedding_function.VectorField()
    text: str = embedding_function.SourceField()
    source: str
    source_type: str  # 'tabular' or 'standard'

# Connect to database
db = lancedb.connect(LANCE_DB_PATH)
table_name = "documents"

# Initialize table cleanly
if table_name in db.table_names():
    table = db.open_table(table_name)
else:
    table = db.create_table(table_name, schema=DocumentChunk, mode="overwrite")

# --- Ingestion Utilities ---

def process_standard_file(file_path: str, filename: str) -> List[str]:
    """Extracts text and splits standard documents (PDF, TXT) into chunks."""
    text_content = ""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content += text + "\n"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported standard file type: {ext}")

    # Simple splitting logic
    words = text_content.split()
    chunk_size = 300  
    overlap = 50
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks

def process_tabular_file(file_path: str, filename: str) -> List[str]:
    """Reads CSV/Excel files and applies our Entity-Anchored Serialization."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".csv":
        df_raw = pd.read_csv(file_path, header=None)
    elif ext in [".xls", ".xlsx"]:
        df_raw = pd.read_excel(file_path, header=None)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported tabular file type: {ext}")

    # Sweep first 15 rows to find the true header
    head_df = df_raw.head(15).fillna("")
    max_non_empty = -1
    header_idx = 0
    
    for idx, row in head_df.iterrows():
        non_empty_count = sum(1 for val in row if str(val).strip() != "")
        if non_empty_count > max_non_empty:
            max_non_empty = non_empty_count
            header_idx = idx

    chunks = []
    if max_non_empty == 0:
        return chunks

    # Re-read with the correct header index
    if ext == ".csv":
        df = pd.read_csv(file_path, header=header_idx)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, header=header_idx)
        
    df = df.fillna("")
    columns = df.columns.tolist()

    if len(columns) == 0:
        return chunks

    primary_col = columns[0]  # Assume 1st column is the primary entity anchor

    for idx, row in df.iterrows():
        entity_name = str(row[primary_col]).strip()
        if not entity_name:
            entity_name = f"Row #{idx + 1}"

        # Construct semantic facts repeating the entity name to boost retrieval
        facts = []
        for col in columns[1:]:
            val = str(row[col]).strip()
            if val:
                facts.append(f"The {col} of {entity_name} is {val}.")
        
        if facts:
            anchored_sentence = f"Regarding {primary_col} {entity_name}: " + " ".join(facts)
            chunks.append(anchored_sentence)

    return chunks

# --- API Endpoints ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Parses files and embeds them cleanly into LanceDB."""
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        ext = os.path.splitext(file.filename)[1].lower()
        is_tabular = ext in [".csv", ".xls", ".xlsx"]
        source_type = "tabular" if is_tabular else "standard"

        if is_tabular:
            chunks = process_tabular_file(temp_file_path, file.filename)
        else:
            chunks = process_standard_file(temp_file_path, file.filename)

        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted.")

        data_to_insert = [
            {
                "text": chunk,
                "source": file.filename,
                "source_type": source_type
            }
            for chunk in chunks
        ]

        global table
        table.add(data_to_insert)
        table.create_fts_index("text", replace=True)

        return {
            "status": "success",
            "filename": file.filename,
            "source_type": source_type,
            "chunks_processed": len(chunks)
        }

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


class ChatRequest(BaseModel):
    query: str
    source_type: str  # 'tabular' or 'standard'

@app.post("/chat")
async def chat(request: ChatRequest):
    """Performs explicit query embedding to avoid registry bugs, then queries Ollama."""
    if request.source_type not in ["tabular", "standard"]:
        raise HTTPException(status_code=400, detail="source_type must be 'tabular' or 'standard'")

    # Configure hybrid search ratio (alpha) dynamically
    alpha = 0.3 if request.source_type == "tabular" else 0.6

    try:
        # Step 1: Explicitly generate the embedding for the query text
        query_vector = embedding_function.compute_query_embeddings(request.query)[0]

        # Step 2: Use explicit linear combiner to merge vectors and full-text search
        reranker = LinearCombinationReranker(weight=alpha)

        search_results = (
            table.search(query_type="hybrid")
            .vector(query_vector)
            .text(request.query)
            .where(f"source_type = '{request.source_type}'", prefilter=True)
            .limit(5)
            .rerank(reranker=reranker)
            .to_list()
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Database search failed: {str(e)}")

    if not search_results:
        return {"response": "I couldn't find any relevant context in the uploaded documents.", "sources": []}

    context_str = "\n".join([f"- {res['text']}" for res in search_results])

    system_prompt = (
        "You are an offline assistant answering questions based strictly on the provided document context.\n"
        "If you do not know the answer or if it's not present in the context, say that you do not know.\n"
        "Do not make up facts.\n\n"
        f"--- DOCUMENT CONTEXT ---\n{context_str}\n\n"
        f"--- USER QUESTION ---\n{request.query}"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": system_prompt,
                "stream": False
            },
            timeout=30
        )
        llm_response = response.json().get("response", "No response generated.")
    except Exception as e:
        llm_response = f"[Error connecting to Ollama: {str(e)}]."

    return {
        "response": llm_response,
        "sources": [
            {"text": res["text"], "source": res["source"], "score": res.get("_score", None)}
            for res in search_results
        ]
    }