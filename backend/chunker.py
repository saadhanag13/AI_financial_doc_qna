# backend/chunker.py -> Improved chunking with better context awareness

import uuid
from typing import List, Dict, Any
import pandas as pd
import re


def _split_text_semantic(text: str, max_words: int = 200, overlap: int = 30) -> List[str]:
    if not text.strip():
        return []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        words = para.split()
        
        if len(current_chunk.split()) + len(words) <= max_words:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            if len(words) > max_words:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                
                for sentence in sentences:
                    sentence_words = sentence.split()
                    if len(temp_chunk.split()) + len(sentence_words) <= max_words:
                        temp_chunk += (" " if temp_chunk else "") + sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sentence
                
                current_chunk = temp_chunk
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            prev_words = chunks[i-1].split()
            if len(prev_words) >= overlap:
                overlap_text = " ".join(prev_words[-overlap:])
                chunk = overlap_text + " " + chunk
        overlapped_chunks.append(chunk)
    
    return overlapped_chunks


def _create_table_summary(csv_str: str, table_id: str, doc_id: str = None, page_no: int = None, filename: str = None) -> str:
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_str))
        
        if df.empty:
            return ""
        
        summary_parts = []
        summary_parts.append(f"Table {table_id} contains {len(df)} rows and {len(df.columns)} columns.")
        
        columns = ", ".join(df.columns.tolist())
        summary_parts.append(f"Columns: {columns}")
        
        sample_rows = min(3, len(df))
        summary_parts.append(f"Sample data (first {sample_rows} rows):")
        
        for idx in range(sample_rows):
            row_data = []
            for col in df.columns:
                value = df.iloc[idx][col]
                row_data.append(f"{col}: {value}")
            summary_parts.append(f"Row {idx + 1}: {', '.join(row_data)}")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append("Numerical summaries:")
            for col in numeric_cols:
                if not df[col].isna().all():
                    total = df[col].sum()
                    avg = df[col].mean()
                    summary_parts.append(f"{col}: Total={total:.2f}, Average={avg:.2f}")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        return f"Table {table_id}: Unable to parse table data - {str(e)}"


def _should_include_chunk(content: str, min_length: int = 20) -> bool:
    if len(content.strip()) < min_length:
        return False
    
    text_chars = sum(1 for c in content if c.isalpha())
    if text_chars < len(content) * 0.3: 
        return False
    
    return True

def _create_table_row_chunks(csv_str: str, table_id: str, doc_id: str, page_no: int, filename: str) -> List[Dict[str, Any]]:
    """
    Parse CSV into dataframe and produce:
      - one 'table_summary' chunk (brief)
      - multiple 'table_row' chunks (row-level labeled text)
      - focused 'key_value' chunks for financial terms if found (Net income, Total assets etc.)
    """
    from io import StringIO
    row_chunks = []
    try:
        df = pd.read_csv(StringIO(csv_str))
    except Exception:
        # fallback: produce a simple summary chunk
        return [{
            "chunk_id": uuid.uuid4().hex,
            "doc_id": doc_id,
            "page_number": page_no,
            "chunk_type": "table_summary",
            "content": f"Table {table_id}: Could not parse CSV table.",
            "metadata": {"source": filename, "page_number": page_no, "table_id": table_id}
        }]

    # table_summary
    summary_parts = [
        f"Table {table_id} contains {len(df)} rows and {len(df.columns)} columns.",
        "Columns: " + ", ".join(df.columns.tolist())
    ]
    table_summary = "\n".join(summary_parts)
    row_chunks.append({
        "chunk_id": uuid.uuid4().hex,
        "doc_id": doc_id,
        "page_number": page_no,
        "chunk_type": "table_summary",
        "content": table_summary,
        "metadata": {"source": filename, "page_number": page_no, "table_id": table_id}
    })

    # row-level chunks
    for idx, row in df.iterrows():
        # build "Col1: val1; Col2: val2; ..." text
        parts = []
        for col in df.columns:
            val = row.get(col, "")
            val_str = str(val) if not pd.isna(val) else ""
            parts.append(f"{col}: {val_str}")
        row_text = "; ".join(parts)
        if _should_include_chunk(row_text, min_length=5):
            row_chunks.append({
                "chunk_id": uuid.uuid4().hex,
                "doc_id": doc_id,
                "page_number": page_no,
                "chunk_type": "table_row",
                "content": row_text,
                "metadata": {
                    "source": filename,
                    "page_number": page_no,
                    "table_id": table_id,
                    "row_index": int(idx)
                }
            })

    # key-value detection: look for financial terms in columns and create explicit KV chunks
    financial_terms = ["net income", "net profit", "total assets", "total liabilities", "revenue", "sales", "profit"]
    lower_cols = [c.lower() for c in df.columns]
    for term in financial_terms:
        # match column names containing term
        matches = [i for i, col in enumerate(lower_cols) if term in col]
        if matches:
            for idx in range(len(df)):
                # take matched columns in that row
                kv_parts = []
                for mi in matches:
                    col_name = df.columns[mi]
                    val = df.iloc[idx, mi]
                    if pd.notna(val) and str(val).strip():
                        kv_parts.append(f"{col_name}: {val}")
                if kv_parts:
                    kv_text = f"{term.title()} (detected): " + "; ".join(kv_parts)
                    row_chunks.append({
                        "chunk_id": uuid.uuid4().hex,
                        "doc_id": doc_id,
                        "page_number": page_no,
                        "chunk_type": "table_kv",
                        "content": kv_text,
                        "metadata": {
                            "source": filename,
                            "page_number": page_no,
                            "table_id": table_id,
                            "row_index": int(idx),
                            "kv_term": term
                        }
                    })

    return row_chunks


def chunk_document(doc: Dict[str, Any], max_words: int = 200, overlap: int = 30) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    doc_id = doc.get("doc_id")
    filename = doc.get("filename", "unknown")
    
    for page in doc.get("pages", []):
        page_no = page.get("page_number", 1)
        
        # --- handle text ---
        text = page.get("text", "").strip()
        if text:
            text_chunks = _split_text_semantic(text, max_words=max_words, overlap=overlap)
            for i, chunk_text in enumerate(text_chunks):
                if _should_include_chunk(chunk_text):
                    chunks.append({
                        "chunk_id": uuid.uuid4().hex,
                        "doc_id": doc_id,
                        "page_number": page_no,
                        "chunk_type": "text",
                        "content": chunk_text,
                        "metadata": {
                            "source": filename,
                            "page_number": page_no,
                            "chunk_index": i,
                            "word_count": len(chunk_text.split())
                        }
                    })
        
        # --- handle tables ---
        tables = page.get("tables", [])
        for table in tables:
            table_id = table.get("table_id")
            csv_str = table.get("csv", "")
            
            if csv_str.strip():
                table_chunks = _create_table_row_chunks(csv_str, table_id, doc_id, page_no, filename)
                for tc in table_chunks:
                    if _should_include_chunk(tc.get("content", ""), min_length=5):
                        chunks.append(tc)
    
    return chunks


def get_chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks:
        return {"total": 0}
    
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table_summary"]
    
    return {
        "total": len(chunks),
        "text_chunks": len(text_chunks),
        "table_chunks": len(table_chunks),
        "avg_words_per_chunk": sum(len(c["content"].split()) for c in chunks) / len(chunks),
        "pages_processed": len(set(c["page_number"] for c in chunks))
    }


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <doc_json_file>")
        sys.exit(1)
        
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        doc_json = json.load(f)
        
    chunks = chunk_document(doc_json)
    stats = get_chunk_stats(chunks)
    
    print(f"Generated {len(chunks)} chunks")
    print("Statistics:", json.dumps(stats, indent=2))
    print("\nFirst 2 chunks preview:")
    print(json.dumps(chunks[:2], indent=2, ensure_ascii=False))