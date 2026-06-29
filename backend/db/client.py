import lancedb
from config import LANCE_DB_PATH
from backend.db.schemas import DocumentChunk

db = lancedb.connect(LANCE_DB_PATH)
table_name = "documents"

if table_name in db.table_names():
    table = db.open_table(table_name)
else:
    table = db.create_table(table_name, schema=DocumentChunk, mode="overwrite")
