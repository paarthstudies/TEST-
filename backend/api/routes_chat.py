from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
import requests
from lancedb.rerankers import LinearCombinationReranker

from backend.db.client import table
from backend.db.embedder import embedding_function
from backend.core.memory import condense_query
from config import OLLAMA_URL, OLLAMA_MODEL

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    source_type: str  # 'tabular' or 'standard'
    chat_history: Optional[List[dict]] = []  # Interactive memory history

@router.post("/chat")
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
