from lancedb.pydantic import LanceModel, Vector
from backend.db.embedder import embedding_function

class DocumentChunk(LanceModel):
    vector: Vector(embedding_function.ndims()) = embedding_function.VectorField()
    text: str = embedding_function.SourceField()  # Always use SourceField to avoid registry resolution errors
    source: str
    source_type: str  # 'tabular' or 'standard'
    sheet_name: str = ""  # Multi-tab excel support metadata
