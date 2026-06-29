from lancedb.embeddings import get_registry
from config import EMBEDDING_MODEL_NAME

registry = get_registry().get("sentence-transformers")
embedding_function = registry.create(name=EMBEDDING_MODEL_NAME, device="cpu")
