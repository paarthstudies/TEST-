# Paths
DOCUMENT_FOLDER   = "./Docs/"
FAISS_INDEX_PATH  = "./faiss_store/index.faiss"
FAISS_META_PATH   = "./faiss_store/metadata.json"   # maps int index → {text, source, page}
BM25_PICKLE_PATH  = "./faiss_store/bm25_chunks.pkl"

# Embedding
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama
OLLAMA_BASE_URL   = "http://localhost:11434/v1"
OLLAMA_API_KEY    = "ollama"                         # required by openai-compatible client, value is arbitrary
GENERATOR_MODEL   = "llama3.1:8b"
JUDGE_MODEL       = "ggozad/prometheus2:latest"

# Chunking
CHUNK_SIZE        = 1500    # characters
CHUNK_OVERLAP     = 150     # characters

# Retrieval
VECTOR_K          = 5       # top-k from FAISS
BM25_K            = 5       # top-k from BM25
ALPHA_VECTOR      = 0.6     # RRF weight for vector results
ALPHA_BM25        = 0.4     # RRF weight for BM25 results
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CONTEXT_TOP_N     = 3       # chunks passed to LLM after reranking

# Generation
TEMPERATURE       = 0.0
