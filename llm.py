"""
llm.py – Thin wrapper around the local Ollama LLM for RAG generation.

The Ollama daemon must be running locally (`ollama serve`) and the chosen
model must already be pulled (`ollama pull llama3`).

This module is intentionally slim: it constructs the RAG prompt and streams
(or returns) the completion.  All retrieval logic lives in main.py / database.py.
"""

from __future__ import annotations

import logging
from typing import List

import ollama

from config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama client (configured once at module import)
# ---------------------------------------------------------------------------

_client = ollama.Client(host=OLLAMA_HOST)


# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

_RAG_PROMPT_TEMPLATE = """\
You are a precise, factual assistant. Answer the user's question using ONLY \
the information provided in the context below.

If the answer cannot be found in the context, respond with:
"I could not find the answer in the provided documents."

Do not speculate, infer, or use any external knowledge.

─── CONTEXT ────────────────────────────────────────────────────────────────
{context}
────────────────────────────────────────────────────────────────────────────

USER QUESTION: {question}

ANSWER:"""


def build_rag_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Construct the strict RAG prompt by injecting retrieved context chunks.

    Each chunk is numbered and separated so the LLM can clearly distinguish
    individual source passages.

    Args:
        question:       The user's raw query.
        context_chunks: List of retrieved text strings (already ranked).

    Returns:
        The fully assembled prompt string ready for the LLM.
    """
    numbered_chunks = "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    return _RAG_PROMPT_TEMPLATE.format(
        context=numbered_chunks,
        question=question,
    )


def generate_answer(question: str, context_chunks: List[str]) -> str:
    """
    Send the RAG prompt to Ollama and return the generated answer.

    Args:
        question:       The user's raw query.
        context_chunks: Top-K retrieved text chunks to inject as context.

    Returns:
        The LLM's answer string.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    prompt = build_rag_prompt(question, context_chunks)

    logger.info(
        "Sending RAG prompt to Ollama (model=%s, context_chunks=%d).",
        OLLAMA_MODEL,
        len(context_chunks),
    )

    try:
        response = _client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            # Keep generation deterministic for factual RAG answers
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1024,   # max output tokens
            },
        )
    except Exception as exc:
        logger.error("Ollama generation failed: %s", exc)
        raise RuntimeError(
            f"Failed to get a response from Ollama ({OLLAMA_MODEL}). "
            f"Make sure `ollama serve` is running and the model is pulled. "
            f"Original error: {exc}"
        ) from exc

    answer: str = response.get("response", "").strip()
    logger.info("Ollama response received (%d chars).", len(answer))
    return answer
