# 🧠 AI Financial Document QnA System

A comprehensive Retrieval-Augmented Generation (RAG) system for analyzing financial documents, reports, and business statements. Built with FastAPI backend, Streamlit frontend, and powered by local LLM through Ollama.

---

## 🚀 Features Implemented

- **Document ingestion & processing**
  - Extract text & tables from PDFs
  - Chunking with overlap for context preservation
  - Metadata tracking (doc_id, page number, source)

- **Embeddings + Retrieval**
  - FAISS vector database for semantic search
  - Keyword + quality-based re-ranking
  - Redundancy filtering for concise context

- **RAG Pipeline**
  - `backend/qa.py` orchestrates retrieval + LLM
  - Specialized **financial prompts** to extract:
    - Balance Sheet, Income Statement, Cash Flow, Retained Earnings
    - Monetary values ($, %, dates)
    - Statement references (page + section)

- **LLM Integration (Ollama)**
  - Local inference via [Ollama](https://ollama.ai/)
  - Tested with `llama2`, `mistral`, `finance-custom` models
  - System + user prompt engineering for structured answers

- **Backend API (FastAPI)**
  - `/` → Health check  
  - `GET /ask?query=...` → Simple QA (direct Ollama call)  
  - `POST /ask` → Full **RAG QA pipeline** with provenance

- **Frontend**
  - `frontend/app.py` (Streamlit prototype)
  - UI for uploading PDFs and asking questions
  - Shows answers + provenance (doc_id, page number, preview)

- **Testing**
  - `tests/pipeline.py` for unit tests:
    - Chunking
    - Embedding
    - Retrieval
    - Document processing

---

## 📂 Project Structure

AI_financial_doc_qna/
│── backend/
│ ├── app.py # FastAPI app
│ ├── qa.py # RAG QA pipeline
│ ├── retriever.py # Retrieval + reranking
│ ├── embedder.py # Embedding + FAISS index
│ ├── processor.py # PDF/text ingestion
│ ├── chunker.py # Chunk splitting logic
│ ├── ollama_client.py # Wrapper for local Ollama API
│── db/
│ ├── db_utils.py # SQLite / FAISS index management
│── frontend/
│ ├── app.py # Streamlit UI (optional)
│── tests/
│ ├── pipeline.py # Unit tests for core pipeline
│── requirements.txt
│── README.md

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Ollama - Install from ollama.ai
- System Dependencies (for PDF/OCR processing):

### Clone the repo
```bash
git clone https://github.com/saadhanag13/AI_financial_doc_qna.git
cd AI_financial_doc_qna
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Setup Ollama and download model:
``` bash
# Install Ollama (if not already done)
# Then download a model:
ollama pull llama3.1
# or
ollama pull llama3
```

### Verify system dependencies:
```bash
python scripts/setup_env.py
```

### Configuration 
```bash
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
MODEL_NAME=llama3.1

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG_MODE=true

# Paths
DATA_DIR=./data
DB_PATH=./data/app.db
```

### Usage:
### Starting the Services:
- Start ollama: 
```bash
ollama serve
```

- Start the Backend API:
```bash
cd backend
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

- Start the Frontend:
```bash
cd streamlit run frontend/app.py
```

- Access the application:
```bash 
Frontend: http://localhost:8501
API Documentation: http://localhost:8000/docs
```
---

## API Endpoints
### Core Endpoints

- POST /ask - Main Q&A with document context
- POST /ask-document - Query specific document by filename
- GET /documents - List uploaded documents
- GET /search - Search chunks with filters
- POST /generate - Direct LLM generation (no RAG)

---

## Utility Endpoints

- GET /health - System health check
- GET /stats - System statistics
- GET /test-context - Test document context detection

---

## Testing- Run Unit tests
```bash
python tests/pipeline/py
```

---

## Troubleshooting
### Ollama not responding
- Ensure Ollama is running: ollama serve
- Check model is downloaded: ollama list

### "ystem dependencies missing
- Run: python scripts/setup_env.py
- Install missing OCR/PDF tools

### Database errors
- Run: python scripts/fix_database_schema.py
- Check data/ directory permissions

### Poor answer quality
- Verify document upload and processing
- Check chunk quality in /search endpoint
- Adjust chunking parameters in chunker.py

---

## 🧑‍💻 Author
### Saadhana Ganesa Narasimhan
MSc Graduate | Aspiring AI/ML Engineer | Passionate about real-world deep learning applications

### 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://saadhanag13.github.io/MyResume/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/saadhana-ganesh-45a50a18b/)
