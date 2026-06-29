import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.core.parsers import process_tabular_file, process_standard_file
from backend.db.client import table

router = APIRouter()

@router.post("/upload")
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
