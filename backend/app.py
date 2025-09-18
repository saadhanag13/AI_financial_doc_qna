#backend/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import logging

from backend.ollama_client import generate
from backend.qa import answer_question, answer_financial_question
from backend import retriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Financial Doc QnA Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QuestionRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    include_debug: Optional[bool] = False

class DirectGenerateRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 512

class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: Optional[str] = None
    quality_score: Optional[float] = None
    sources_used: int
    provenance: Optional[List[Dict[str, Any]]] = None
    debug_info: Optional[List[str]] = None

class DirectResponse(BaseModel):
    query: str
    response: str
    model_used: str


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "message": "AI Financial Doc QnA Backend Running ✅",
        "version": "1.0.0"
    }
    
    
@app.post("/ask")
async def ask_question(query: str, target_doc_id: str = None):
    answer, provenance, history = answer_question(query, target_doc_id=target_doc_id)
    return {"question": query, "answer": answer, "provenance": provenance}


@app.get("/health")
def detailed_health():
    try:
        from db import utils
        docs = utils.list_documents()
        doc_count = len(docs)
        
        from backend import embedder
        embedder._init_faiss()
        
        return {
            "status": "healthy",
            "database": "connected",
            "embedding_system": "loaded",
            "documents_indexed": doc_count,
            "model": os.getenv("MODEL_NAME", "llama3:latest")
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question_rag(request: QuestionRequest):

    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Processing RAG question: {query}")
        
        answer, provenance, debug_history = answer_question(
            question=query,
            top_k=request.top_k,
            include_debug=request.include_debug
        )
        
        chunks = retriever.retrieve(query, top_k=request.top_k)
        quality_score = answer_financial_question(answer, chunks)
        
        confidence = "high"
        if quality_score < 0.3:
            confidence = "low"
        elif quality_score < 0.6:
            confidence = "medium"
        
        response = AnswerResponse(
            question=query,
            answer=answer,
            confidence=confidence,
            quality_score=quality_score,
            sources_used=len(provenance),
            provenance=provenance if os.getenv("DEBUG_MODE", "false").lower() == "true" else None,
            debug_info=debug_history if request.include_debug else None
        )
        
        logger.info(f"RAG response generated with {len(provenance)} sources")
        return response
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/generate", response_model=DirectResponse)
async def direct_generate(request: DirectGenerateRequest):

    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Processing direct generation: {query}")
        
        messages = [{"role": "user", "content": query}]
        response_text = generate(messages, max_tokens=request.max_tokens)
        
        response = DirectResponse(
            query=query,
            response=response_text,
            model_used=os.getenv("MODEL_NAME", "llama3:latest")
        )
        
        logger.info("Direct generation completed")
        return response
        
    except Exception as e:
        logger.error(f"Error in direct generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/documents")
def list_documents():
    try:
        from db import utils
        docs = utils.list_documents()
        return {
            "count": len(docs),
            "documents": docs
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/search")
def search_chunks(query: str, top_k: int = 5):
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        chunks = retriever.retrieve(query, top_k=top_k)
        stats = retriever.get_retrieval_stats(chunks)
        
        return {
            "query": query,
            "chunks_found": len(chunks),
            "chunks": chunks,
            "retrieval_stats": stats
        }
    except Exception as e:
        logger.error(f"Error in chunk search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/stats")
def get_system_stats():
    try:
        from db import utils
        import sqlite3
        
        docs = utils.list_documents()
        
        with sqlite3.connect("data/app.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type")
            chunk_types = dict(cursor.fetchall())
        
        return {
            "documents": len(docs),
            "total_chunks": chunk_count,
            "chunk_types": chunk_types,
            "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "llm_model": os.getenv("MODEL_NAME", "llama3:latest"),
            "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.get("/ask")
def ask_get_legacy(query: str):
    try:
        messages = [{"role": "user", "content": query}]
        response_text = generate(messages)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error in legacy endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )
