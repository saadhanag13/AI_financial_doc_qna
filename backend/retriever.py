#backend/retriever.py

# backend/retriever.py - QUICK FIX VERSION

import re
from typing import List, Dict, Any, Set, Optional
import math
from typing import Optional

from backend import embedder


def extract_query_keywords(query: str) -> Set[str]:
    stop_words = {"what", "is", "the", "was", "were", "are", "in", "on", "at", "to", "for", "of", "with", "by"}
    
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = set()
    
    for word in words:
        if len(word) > 2 and word not in stop_words:
            keywords.add(word)
    
    numbers = re.findall(r'\b\d{4}\b|\b\d+\b', query)  
    keywords.update(numbers)
    
    return keywords


def calculate_keyword_score(content: str, keywords: Set[str]) -> float:
    if not keywords:
        return 0.0
    
    content_lower = content.lower()
    content_words = re.findall(r'\b\w+\b', content_lower)
    
    if not content_words:
        return 0.0
    
    keyword_counts = {}
    for keyword in keywords:
        count = len(re.findall(rf'\b{re.escape(keyword)}\b', content_lower))
        if count > 0:
            keyword_counts[keyword] = count
    
    if not keyword_counts:
        return 0.0
    
    total_words = len(content_words)
    score = 0.0
    
    for keyword, count in keyword_counts.items():
        tf = count / total_words
        idf = math.log(1 + 1/max(count, 1))
        
        boost = 1.0
        if keyword.isdigit() and len(keyword) == 4:  
            boost = 1.5
        elif keyword in ['revenue', 'profit', 'loss', 'total', 'fy', 'fiscal', 'assets', 'income', 'net']:
            boost = 1.3
        
        score += tf * idf * boost
    
    return score


def calculate_chunk_quality_score(chunk: Dict[str, Any]) -> float:
    content = chunk.get("content", "")
    chunk_type = chunk.get("chunk_type", "text")
    
    base_score = 1.0
    
    if chunk_type == "table_summary":
        base_score *= 1.2
    
    numbers = len(re.findall(r'\d+', content))
    if numbers > 0:
        base_score *= 1 + (numbers * 0.1)
    
    word_count = len(content.split())
    if word_count > 50:
        base_score *= 1.1
    elif word_count < 20:
        base_score *= 0.9
    
    # Enhanced financial terms
    financial_terms = ['revenue', 'profit', 'loss', 'margin', 'expense', 'cost', 'total', 'assets', 'liabilities', 'income', 'net']
    financial_score = sum(1 for term in financial_terms if term in content.lower())
    base_score *= 1 + (financial_score * 0.05)
    
    return base_score


def filter_chunks_by_document(chunks: List[Dict[str, Any]], target_doc_id: str) -> List[Dict[str, Any]]:
    """Filter chunks to only include those from a specific document"""
    if not target_doc_id:
        return chunks
    
    filtered = []
    for chunk in chunks:
        chunk_doc_id = chunk.get("metadata", {}).get("doc_id")
        if chunk_doc_id == target_doc_id:
            filtered.append(chunk)
    
    return filtered


def rerank_chunks(chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    keywords = extract_query_keywords(query)
    
    scored_chunks = []
    for chunk in chunks:
        content = chunk.get("content", "")
        
        similarity_score = chunk.get("score", 0)
        keyword_score = calculate_keyword_score(content, keywords)
        quality_score = calculate_chunk_quality_score(chunk)
        
        final_score = (
            similarity_score * 0.5 +     
            keyword_score * 0.3 +        
            quality_score * 0.2           
        )
        
        chunk_copy = chunk.copy()
        chunk_copy["final_score"] = final_score
        chunk_copy["keyword_score"] = keyword_score
        chunk_copy["quality_score"] = quality_score
        
        scored_chunks.append(chunk_copy)
    
    scored_chunks.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_chunks


def filter_redundant_chunks(chunks: List[Dict[str, Any]], similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    if len(chunks) <= 1:
        return chunks
    
    filtered = [chunks[0]]  
    
    for chunk in chunks[1:]:
        content = chunk.get("content", "").lower()
        
        is_redundant = False
        for selected in filtered:
            selected_content = selected.get("content", "").lower()
            
            words1 = set(content.split())
            words2 = set(selected_content.split())
            
            if words1 and words2:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                jaccard_sim = intersection / union
                
                if jaccard_sim > similarity_threshold:
                    is_redundant = True
                    break
        
        if not is_redundant:
            filtered.append(chunk)
            
        if len(filtered) >= 5:
            break
    
    return filtered


def retrieve(query: str, top_k: int = 5, use_reranking: bool = True, filter_redundant: bool = True, 
            target_doc_id: Optional[str] = None, use_document_context: bool = True) -> List[Dict[str, Any]]:
    """
    FIXED: Added the missing parameters that qa.py expects
    """
    if not query or not query.strip():
        return []
    
    # Get more initial results for reranking
    initial_k = min(top_k * 5, 50)  
    ann_results = embedder.query_index(query, top_k=initial_k)
    
    # if not ann_results:
    #     return []
    
    # Convert to standardized format
    chunks = []
    if ann_results:
        for chunk_id, distance, metadata in ann_results:
            similarity_score = 1 / (1 + distance) 
            
            chunk = {
                "chunk_id": chunk_id,
                "score": similarity_score,
                "content": metadata.get("content", "") if isinstance(metadata, dict) else "",
                "chunk_type": metadata.get("chunk_type", "text") if isinstance(metadata, dict) else "text",
                "metadata": {
                    "doc_id": metadata.get("doc_id") if isinstance(metadata, dict) else None,
                    "page_number": metadata.get("page_number") if isinstance(metadata, dict) else None,
                    "source": metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown",
                }
            }
        chunks.append(chunk)
    
    # Filter by target document if specified
    if target_doc_id:
        chunks = [c for c in chunks if c.get("metadata", {}).get("doc_id") == target_doc_id]    
    
    if not chunks:
        try:
            # embedder._get_conn exists in your embedder; use it to query chunks table directly
            with embedder._get_conn() as conn:
                # Construct a simple LIKE pattern based on the raw query and important keywords
                keywords = list(extract_query_keywords(query))
                patterns = set()
                # prefer multi-word query match first
                if len(query.strip()) > 3:
                    patterns.add(f"%{query.strip()}%")
                # add single-word keyword patterns
                for kw in keywords:
                    patterns.add(f"%{kw}%")
                rows = []
                for p in patterns:
                    cur = conn.execute(
                        "SELECT chunk_id, doc_id, page_number, chunk_type, content, metadata FROM chunks WHERE content LIKE ? LIMIT ?",
                        (p, top_k * 5)
                    )
                    rows.extend(cur.fetchall())

                # deduplicate row by chunk_id
                seen = set()
                for r in rows:
                    cid, docid, pno, ctype, content, metadata = r
                    if cid in seen:
                        continue
                    seen.add(cid)
                    chunks.append({
                        "chunk_id": cid,
                        "score": 0.5,  # fallback baseline score
                        "content": content,
                        "chunk_type": ctype,
                        "metadata": {"doc_id": docid, "page_number": pno, "source": getattr(metadata, 'source', None) or metadata}
                    })
        except Exception:
            # If DB fallback fails, just continue with empty chunks
            pass

    # Re-rank chunks
    if use_reranking and len(chunks) > 1:
        chunks = rerank_chunks(chunks, query)
    
    # Filter redundant chunks
    if filter_redundant and len(chunks) > 1:
        chunks = filter_redundant_chunks(chunks)
    
    return chunks[:top_k]
    


def get_retrieval_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks:
        return {"total": 0}
    
    stats = {
        "total": len(chunks),
        "avg_score": sum(c.get("score", 0) for c in chunks) / len(chunks),
        "score_range": [
            min(c.get("score", 0) for c in chunks),
            max(c.get("score", 0) for c in chunks)
        ],
        "chunk_types": {}
    }
    
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "unknown")
        stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1
    
    sources = [c.get("metadata", {}).get("source", "unknown") for c in chunks]
    stats["unique_sources"] = len(set(sources))
    
    return stats


def batch_retrieve(queries: List[str], top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    results = {}
    
    for query in queries:
        chunks = retrieve(query, top_k=top_k)
        results[query] = chunks
    
    return results


if __name__ == "__main__":
    # Test with financial query
    test_query = "what was the net income in the service income statement?"
    print(f"Testing: {test_query}")
    
    chunks = retrieve(test_query, top_k=3, use_reranking=True)
    
    print(f"Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("final_score", chunk.get("score", 0))
        content_preview = chunk.get("content", "")[:100] + "..."
        source = chunk.get("metadata", {}).get("source", "unknown")
        print(f"  {i}. Score: {score:.3f} | Source: {source}")
        print(f"     Content: {content_preview}")
    
    stats = get_retrieval_stats(chunks)
    print(f"Stats: {stats}")