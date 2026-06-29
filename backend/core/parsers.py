import os
from typing import List
from fastapi import HTTPException
from pypdf import PdfReader
import pandas as pd

def process_standard_file(file_path: str, filename: str) -> List[str]:
    """Extracts text and splits standard documents (PDF, TXT) into chunks."""
    text_content = ""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content += text + "\n"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported standard file type: {ext}")

    # Simple splitting logic
    words = text_content.split()
    chunk_size = 300  
    overlap = 50
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks

def process_tabular_file(file_path: str, filename: str) -> List[dict]:
    """Reads CSV/Excel files (supporting multi-tab Excel sheets),
    dynamically strips title blocks/empty headers per sheet,
    and applies our Entity-Anchored Serialization."""
    ext = os.path.splitext(filename)[1].lower()
    
    sheets_data = {}
    if ext == ".csv":
        try:
            df = pd.read_csv(file_path, header=None)
            sheets_data[""] = df  # CSV has no tab name, default to empty string
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")
    elif ext in [".xls", ".xlsx"]:
        try:
            # Read all sheets; returns dict of {sheet_name: df}
            sheets_data = pd.read_excel(file_path, sheet_name=None, header=None)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read Excel sheets: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported tabular file type: {ext}")

    all_chunks = []

    for sheet_name, df in sheets_data.items():
        if df is None or df.empty:
            continue

        # --- Messy Spreadsheet Cleanup Heuristic ---
        # Look at the first 15 rows and find the one containing the maximum non-empty cells.
        header_row_idx = 0
        max_non_empty = 0
        for i in range(min(15, len(df))):
            non_empty_count = df.iloc[i].notna().sum()
            if non_empty_count > max_non_empty:
                max_non_empty = non_empty_count
                header_row_idx = i

        # Extract the header row and clean column names
        raw_headers = df.iloc[header_row_idx].tolist()
        headers = []
        for idx, h in enumerate(raw_headers):
            header_str = str(h).strip() if pd.notna(h) else ""
            if header_str != "":
                headers.append(header_str)
            else:
                headers.append(f"Column_{idx}")

        # Slice the dataframe from the header row downwards
        sheet_df = df.iloc[header_row_idx + 1:].copy()
        sheet_df.columns = headers

        # Clean up empty padding rows & structural columns
        sheet_df = sheet_df.dropna(how="all")
        sheet_df = sheet_df.dropna(how="all", axis=1)
        # Remove entirely empty unnamed columns
        sheet_df = sheet_df.loc[:, [col for col in sheet_df.columns if not col.startswith("Column_") or sheet_df[col].astype(str).str.strip().any()]]
        sheet_df = sheet_df.fillna("")

        columns = sheet_df.columns.tolist()
        if len(columns) == 0:
            continue

        primary_col = columns[0]

        for idx, row in sheet_df.iterrows():
            entity_name = str(row[primary_col]).strip()
            if not entity_name:
                row_vals = [str(row[c]).strip() for c in columns if str(row[c]).strip()]
                entity_name = row_vals[0] if row_vals else f"Row #{idx + 1}"

            # Construct semantic facts repeating entity name with sheet context
            facts = []
            for col in columns[1:]:
                val = str(row[col]).strip()
                if val:
                    if sheet_name:
                        facts.append(f"The {col} of {entity_name} in sheet '{sheet_name}' is {val}.")
                    else:
                        facts.append(f"The {col} of {entity_name} is {val}.")
            
            sheet_prefix = f"Regarding sheet '{sheet_name}', " if sheet_name else ""
            anchored_sentence = f"{sheet_prefix}under column {primary_col} '{entity_name}': " + " ".join(facts)
            
            all_chunks.append({
                "text": anchored_sentence,
                "source_type": "tabular",
                "sheet_name": sheet_name
            })

    return all_chunks
