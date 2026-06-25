"""Query pipeline and answer generation for the local RAG chatbot.

Loads FAISS, BM25, embeddings, reranker, and Ollama clients at startup, then
runs hybrid retrieval, cross-encoder reranking, Llama generation, and Prometheus judging.
"""

import json
import os
import pickle
import sys

import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from sentence_transformers import CrossEncoder

from config import (
    ALPHA_BM25,
    ALPHA_VECTOR,
    BM25_K,
    BM25_PICKLE_PATH,
    CONTEXT_TOP_N,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    RERANKER_MODEL,
    TEMPERATURE,
    VECTOR_K,
)

RRF_K = 60
RRF_TOP_N = 10


def _load_stores():
    missing = [
        path
        for path in (FAISS_INDEX_PATH, FAISS_META_PATH, BM25_PICKLE_PATH)
        if not os.path.isfile(path)
    ]
    if missing:
        print("Error: index files not found. Missing:")
        for path in missing:
            print(f"  - {path}")
        print("\nRun ingest.py first to build the indexes.")
        sys.exit(1)

    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(BM25_PICKLE_PATH, "rb") as f:
        bm25_obj, raw_chunks = pickle.load(f)

    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    generator_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    return faiss_index, metadata, bm25_obj, raw_chunks, embeddings_model, reranker, generator_client


faiss_index, metadata, bm25_obj, raw_chunks, embeddings_model, reranker, generator_client = _load_stores()


def _vector_search(query: str) -> list[dict]:
    query_vec = embeddings_model.embed_query(query)
    query_np = np.array([query_vec], dtype=np.float32)
    distances, indices = faiss_index.search(query_np, VECTOR_K)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx]
        results.append(
            {
                "text": entry["text"],
                "source": entry.get("source"),
                "page": entry.get("page"),
                "score": float(dist),
                "rank": rank,
            }
        )
    return results


def _bm25_search(query: str) -> list[dict]:
    tokens = query.lower().split()
    scores = bm25_obj.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:BM25_K]

    results = []
    for rank, idx in enumerate(top_indices):
        chunk = raw_chunks[idx]
        results.append(
            {
                "text": chunk["text"],
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "score": float(scores[idx]),
                "rank": rank,
            }
        )
    return results


def _rrf_fusion(vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
    fused: dict[str, dict] = {}

    for item in vector_results:
        text = item["text"]
        rrf = ALPHA_VECTOR * (1.0 / (RRF_K + item["rank"]))
        fused[text] = {
            "text": text,
            "source": item.get("source"),
            "page": item.get("page"),
            "rrf_score": rrf,
        }

    for item in bm25_results:
        text = item["text"]
        rrf = ALPHA_BM25 * (1.0 / (RRF_K + item["rank"]))
        if text in fused:
            fused[text]["rrf_score"] += rrf
        else:
            fused[text] = {
                "text": text,
                "source": item.get("source"),
                "page": item.get("page"),
                "rrf_score": rrf,
            }

    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:RRF_TOP_N]


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:CONTEXT_TOP_N]


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        parts.append("---")
        parts.append(chunk["text"])
        parts.append("---")
    return "\n".join(parts)


def _call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    response = generator_client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _parse_json_response(raw_text: str, fallback: dict) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {**fallback, "parse_error": True}


def answer(query: str) -> dict:
    vector_results = _vector_search(query)
    bm25_results = _bm25_search(query)
    fused = _rrf_fusion(vector_results, bm25_results)
    top_chunks = _rerank(query, fused)

    context_block = _format_context(top_chunks)

    system_prompt = (
        "You are a precise question-answering assistant.\n"
        "Answer ONLY from the provided context. Do not use outside knowledge.\n"
        "If the context does not contain enough information, set \"found\" to false.\n"
        "You MUST respond with valid JSON only — no explanation, no markdown.\n"
        "Schema: {\"answer\": string, \"confidence\": float 0-1, \"found\": boolean}"
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"

    raw_generation = _call_llm(GENERATOR_MODEL, system_prompt, user_prompt)
    generation = _parse_json_response(
        raw_generation,
        {"answer": raw_generation, "confidence": 0.0, "found": False},
    )

    llama_answer = generation.get("answer", "")
    confidence = float(generation.get("confidence", 0.0))
    found = bool(generation.get("found", False))

    judge_system = (
        "You are a strict factual grounding evaluator.\n"
        "You will be given retrieved context and an AI-generated answer.\n"
        "Evaluate whether the answer is fully supported by the context.\n"
        "You MUST respond with valid JSON only.\n"
        "Schema: {\"is_supported\": boolean, \"critique\": string (one sentence)}"
    )
    judge_user = f"Context:\n{context_block}\n\nAnswer to evaluate: {llama_answer}"

    raw_judge = _call_llm(JUDGE_MODEL, judge_system, judge_user)
    judge_result = _parse_json_response(
        raw_judge,
        {"is_supported": False, "critique": raw_judge},
    )

    is_supported = bool(judge_result.get("is_supported", False))
    critique = str(judge_result.get("critique", ""))

    sources = [{"text": c["text"], "source": c.get("source")} for c in top_chunks]

    return {
        "answer": llama_answer,
        "confidence": confidence,
        "found": found,
        "is_supported": is_supported,
        "critique": critique,
        "sources": sources,
    }


if __name__ == "__main__":
    print("RAG chatbot ready. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        result = answer(query)
        print(f"\nAnswer     : {result['answer']}")
        print(f"Confidence : {result['confidence']:.2f}")
        print(f"Found      : {result['found']}")
        print(f"Supported  : {result['is_supported']}")
        print(f"Critique   : {result['critique']}")
        print(f"Sources    : {[s['source'] for s in result['sources']]}\n")
