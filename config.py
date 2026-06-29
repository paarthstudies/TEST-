# --- Configurations ---
LANCE_DB_PATH = "./.lancedb"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # Will default to this model for both condensation and answering
CONDENSE_MODEL_NAME = OLLAMA_MODEL
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
