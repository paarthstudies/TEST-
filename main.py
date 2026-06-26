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
OLLAMA_MODEL = "llama3"  # Will default to this model for both condensation and answering
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Embeddings Setup ---
registry = get_registry().get("sentence-transformers")
embedding_function = registry.create(name=EMBEDDING_MODEL_NAME, device="cpu")

# --- Database Schema ---
class DocumentChunk(LanceModel):
    vector: Vector(embedding_function.ndims()) = embedding_function.VectorField()
    text: str = embedding_function.SourceField()  # Always use SourceField to avoid registry resolution errors
    source: str
    source_type: str  # 'tabular' or 'standard'
    sheet_name: str = ""  # Multi-tab excel support metadata

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

def process_tabular_file(file_path: str, filename: str) -> List[dict]:
    """Reads CSV/Excel files (supporting multi-tab Excel sheets),
    dynamically strips title blocks/empty headers per sheet,
    and applies our Entity-Anchored Serialization."""
    ext = os.path.splitext(filename)[1].lower()
    
    sheets_data = {}
    if ext == ".csv":
        try:
            df = pd.read_csv(file_path, header=None)
            sheets_data[""] = df  # CSV has no tab name, default to empty string
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")
    elif ext in [".xls", ".xlsx"]:
        try:
            # Read all sheets; returns dict of {sheet_name: df}
            sheets_data = pd.read_excel(file_path, sheet_name=None, header=None)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read Excel sheets: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported tabular file type: {ext}")

    all_chunks = []

    for sheet_name, df in sheets_data.items():
        if df is None or df.empty:
            continue

        # --- Messy Spreadsheet Cleanup Heuristic ---
        # Look at the first 15 rows and find the one containing the maximum non-empty cells.
        header_row_idx = 0
        max_non_empty = 0
        for i in range(min(15, len(df))):
            non_empty_count = df.iloc[i].notna().sum()
            if non_empty_count > max_non_empty:
                max_non_empty = non_empty_count
                header_row_idx = i

        # Extract the header row and clean column names
        raw_headers = df.iloc[header_row_idx].tolist()
        headers = []
        for idx, h in enumerate(raw_headers):
            header_str = str(h).strip() if pd.notna(h) else ""
            if header_str != "":
                headers.append(header_str)
            else:
                headers.append(f"Column_{idx}")

        # Slice the dataframe from the header row downwards
        sheet_df = df.iloc[header_row_idx + 1:].copy()
        sheet_df.columns = headers

        # Clean up empty padding rows & structural columns
        sheet_df = sheet_df.dropna(how="all")
        sheet_df = sheet_df.dropna(how="all", axis=1)
        # Remove entirely empty unnamed columns
        sheet_df = sheet_df.loc[:, [col for col in sheet_df.columns if not col.startswith("Column_") or sheet_df[col].astype(str).str.strip().any()]]
        sheet_df = sheet_df.fillna("")

        columns = sheet_df.columns.tolist()
        if len(columns) == 0:
            continue

        primary_col = columns[0]

        for idx, row in sheet_df.iterrows():
            entity_name = str(row[primary_col]).strip()
            if not entity_name:
                row_vals = [str(row[c]).strip() for c in columns if str(row[c]).strip()]
                entity_name = row_vals[0] if row_vals else f"Row #{idx + 1}"

            # Construct semantic facts repeating entity name with sheet context
            facts = []
            for col in columns[1:]:
                val = str(row[col]).strip()
                if val:
                    if sheet_name:
                        facts.append(f"The {col} of {entity_name} in sheet '{sheet_name}' is {val}.")
                    else:
                        facts.append(f"The {col} of {entity_name} is {val}.")
            
            sheet_prefix = f"Regarding sheet '{sheet_name}', " if sheet_name else ""
            anchored_sentence = f"{sheet_prefix}under column {primary_col} '{entity_name}': " + " ".join(facts)
            
            all_chunks.append({
                "text": anchored_sentence,
                "source_type": "tabular",
                "sheet_name": sheet_name
            })

    return all_chunks

# --- Query Condensation Engine ---

def condense_query(chat_history: list, current_query: str) -> str:
    """Uses our local Ollama model to condense conversational query history."""
    if not chat_history:
        return current_query  # Return raw query if no history exists

    # Format history context (limiting to the last 4 messages)
    formatted_history = ""
    for msg in chat_history[-4:]:
        role = "User" if msg.get("role", "user") == "user" else "Assistant"
        content = msg.get("content", "")
        formatted_history += f"{role}: {content}\n"

    system_prompt = (
        "You are an AI query-refactoring engine executing on an offline database.\n"
        "Your task is to analyze the chat history and the follow-up question below, "
        "and rephrase the follow-up question into a standalone, keyword-rich search query.\n"
        "You must resolve all pronouns (e.g. 'he', 'she', 'it', 'their', 'his', 'her') to their "
        "corresponding entity names based on the conversation history.\n"
        "DO NOT answer the question. Return ONLY the rephrased standalone search query.\n\n"
        f"--- CONVERSATION HISTORY ---\n{formatted_history}\n"
        f"--- FOLLOW-UP QUESTION ---\n{current_query}\n\n"
        "Standalone Query:"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": system_prompt,
                "stream": False
            },
            timeout=15
        )
        condensed_query = response.json().get("response", current_query).strip()
        # Clean potential wrapped quotes from Ollama
        condensed_query = condensed_query.strip('"\'')
        return condensed_query
    except Exception as e:
        # Graceful fallback on network timeout
        return current_query

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
            # Chunks are dictionaries with sheet_name metadata populated
            chunks = process_tabular_file(temp_file_path, file.filename)
            data_to_insert = []
            for c in chunks:
                c["source"] = file.filename
                data_to_insert.append(c)
        else:
            raw_chunks = process_standard_file(temp_file_path, file.filename)
            # Default missing sheet_name properties during standard ingestion to prevent validation errors
            data_to_insert = [
                {
                    "text": chunk,
                    "source": file.filename,
                    "source_type": source_type,
                    "sheet_name": ""
                }
                for chunk in raw_chunks
            ]

        if not data_to_insert:
            raise HTTPException(status_code=400, detail="No content extracted.")

        global table
        table.add(data_to_insert)
        table.create_fts_index("text", replace=True)

        return {
            "status": "success",
            "filename": file.filename,
            "source_type": source_type,
            "chunks_processed": len(data_to_insert)
        }

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


class ChatRequest(BaseModel):
    query: str
    source_type: str  # 'tabular' or 'standard'
    chat_history: Optional[List[dict]] = []  # Interactive memory history

@app.post("/chat")
async def chat(request: ChatRequest):
    """Executes query condensation, converts search string to dense vector, and queries Ollama."""
    if request.source_type not in ["tabular", "standard"]:
        raise HTTPException(status_code=400, detail="source_type must be 'tabular' or 'standard'")

    # Step 1: Run conversational memory condensation
    search_query = condense_query(request.chat_history, request.query)

    # Configure hybrid search ratio (alpha) dynamically
    alpha = 0.3 if request.source_type == "tabular" else 0.6

    try:
        # Step 2: Explicitly generate embedding for search query (resolves registry bug)
        query_vector = embedding_function.compute_query_embeddings(search_query)[0]

        # Step 3: Fixed parameter assignment (alpha=alpha instead of weight=alpha)
        reranker = LinearCombinationReranker(alpha)

        search_results = (
            table.search(query_type="hybrid")
            .vector(query_vector)
            .text(search_query)
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
        f"--- USER QUESTION (REPHRASED) ---\n{search_query}"
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