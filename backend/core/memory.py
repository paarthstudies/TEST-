import requests
from config import OLLAMA_URL, OLLAMA_MODEL

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
