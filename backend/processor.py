
#backend/processor.py -> document ingestion

from __future__ import annotations
import os
import uuid
import io
import json
import shutil
import tempfile
from typing import List, Dict, Any, Optional

import pandas as pd

# PDF libraries 
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from pdf2image import convert_from_path, convert_from_bytes
except Exception:
    convert_from_path = None
    convert_from_bytes = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import camelot
except Exception:
    camelot = None

# Utilities
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
INDEXES_DIR = os.path.join(DATA_DIR, "indexes")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(INDEXES_DIR, exist_ok=True)


class ProcessorError(Exception):
    pass


def _save_upload(uploaded_file) -> str:

    filename = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", None) or f"upload_{uuid.uuid4()}"
    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(filename)}"
    out_path = os.path.join(UPLOADS_DIR, safe_name)

    if hasattr(uploaded_file, "read"):
        with open(out_path, "wb") as f:
            uploaded_file.seek(0)
            shutil.copyfileobj(uploaded_file, f)
    else:
        shutil.copyfile(uploaded_file, out_path)

    return out_path


def _is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def _is_excel(path: str) -> bool:
    return path.lower().endswith(('.xls', '.xlsx', '.xlsm', '.xlsb'))


def _guess_scanned(text_pages: List[str], threshold_chars: int = 30) -> bool:

    if not text_pages:
        return True
    low_text_pages = sum(1 for t in text_pages if len(t.strip()) < threshold_chars)
    return (low_text_pages / max(1, len(text_pages))) > 0.5


def _ocr_pdf(path: str) -> List[str]:

    if convert_from_path is None or pytesseract is None:
        raise ProcessorError("OCR requires pdf2image and pytesseract. Please install them and ensure system deps (poppler, tesseract) are available.")

    images = convert_from_path(path)
    page_texts = []
    for img in images:
        text = pytesseract.image_to_string(img)
        page_texts.append(text)
    return page_texts


def _pdf_text_extract(path: str) -> List[str]:

    if pdfplumber is None:
        # Try fitz (PyMuPDF)
        try:
            import fitz
        except Exception:
            raise ProcessorError("pdfplumber or PyMuPDF (fitz) required for PDF text extraction.")
        doc = fitz.open(path)
        pages = [p.get_text("text") for p in doc]
        return pages

    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return pages


def _extract_tables_pdf(path: str) -> Dict[int, List[pd.DataFrame]]:

    tables_by_page: Dict[int, List[pd.DataFrame]] = {}

    if camelot is not None:
        try:
            camelot_tables = camelot.read_pdf(path, pages="all", flavor="stream")
            for t in camelot_tables:
                page_no = int(t.page)
                df = t.df
                tables_by_page.setdefault(page_no, []).append(df)
        except Exception:
            pass

    if pdfplumber is not None:
        try:
            with pdfplumber.open(path) as pdf:
                for i, p in enumerate(pdf.pages, start=1):
                    page_tables = []
                    try:
                        extracted = p.extract_tables()
                        for tab in extracted:
                            df = pd.DataFrame(tab[1:], columns=tab[0]) if tab and len(tab) > 1 else pd.DataFrame(tab)
                            page_tables.append(df)
                    except Exception:
                        pass
                    if page_tables:
                        tables_by_page.setdefault(i, []).extend(page_tables)
        except Exception:
            pass

    return tables_by_page


def _excel_extract(path: str) -> Dict[str, pd.DataFrame]:

    try:
        xls = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        raise ProcessorError(f"Failed to read Excel file: {e}")
    return xls


def _df_to_csv_string(df: pd.DataFrame) -> str:
    try:
        return df.to_csv(index=False)
    except Exception:
        return ""


def process_file(uploaded_file) -> Dict[str, Any]:

    saved_path = _save_upload(uploaded_file)
    filename = os.path.basename(saved_path)
    doc_id = uuid.uuid4().hex

    result = {
        "doc_id": doc_id,
        "filename": filename,
        "path": saved_path,
        "pages": [],
        "metadata": {"original_name": getattr(uploaded_file, "name", filename)}
    }

    try:
        if _is_pdf(saved_path):
            text_pages = _pdf_text_extract(saved_path)

            scanned = _guess_scanned(text_pages)
            if scanned:
                try:
                    ocr_pages = _ocr_pdf(saved_path)
                    text_pages = [op or tp for op, tp in zip(ocr_pages, text_pages + [""] * len(ocr_pages))]
                except Exception as e:
                    print(f"OCR failed: {e}")

            tables_map = _extract_tables_pdf(saved_path)

            num_pages = max(len(text_pages), max(tables_map.keys()) if tables_map else 0)
            for pno in range(1, num_pages + 1):
                page_text = text_pages[pno - 1] if pno - 1 < len(text_pages) else ""
                page_tables = []
                for t_df in tables_map.get(pno, []):
                    table_id = uuid.uuid4().hex
                    page_tables.append({"table_id": table_id, "csv": _df_to_csv_string(t_df)})

                result["pages"].append({"page_number": pno, "text": page_text, "tables": page_tables})

        elif _is_excel(saved_path):
            sheets = _excel_extract(saved_path)
            for i, (sheet_name, df) in enumerate(sheets.items(), start=1):
                table_id = uuid.uuid4().hex
                result["pages"].append({
                    "page_number": i,
                    "sheet_name": sheet_name,
                    "text": "",
                    "tables": [{"table_id": table_id, "csv": _df_to_csv_string(df)}]
                })
        else:
            with open(saved_path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            result["pages"].append({"page_number": 1, "text": txt, "tables": []})

    except Exception as e:
        raise ProcessorError(f"Failed processing file: {e}")

    preview_path = os.path.join(UPLOADS_DIR, f"{doc_id}_preview.json")
    with open(preview_path, "w", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)

    result["preview_path"] = preview_path
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a PDF or Excel file to process")
    args = parser.parse_args()
    print("Processing:", args.path)
    r = process_file(args.path)
    print(json.dumps({"doc_id": r["doc_id"], "filename": r["filename"], "pages": len(r["pages"])}, indent=2))
