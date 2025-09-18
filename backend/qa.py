# backend/qa.py -> RAG QA pipeline

import json
import re
from typing import Dict, Any, List, Tuple, Optional

from backend import retriever
from backend import ollama_client
from db import utils

# Enhanced financial document prompts
FINANCIAL_SYSTEM_PROMPT = """You are a financial document analyst specializing in financial statements, reports, and business documents. Your task is to answer questions using ONLY the provided document content.

CRITICAL RULES:
1. Answer based SOLELY on the provided financial document content
2. For financial questions, always include specific dollar amounts, percentages, and dates
3. Reference the specific financial statement (Income Statement, Balance Sheet, Cash Flow, etc.)
4. Include the reporting period/date when available
5. If asking about financial metrics, provide the exact figures and their context
6. Format monetary amounts clearly (e.g., $13,060, not 13060 or $13K)
7. If the information is not in the provided documents, state: "The provided financial documents do not contain information about [topic]"

FINANCIAL STATEMENT TYPES:
- Income Statement: Revenue, expenses, net income, operating income
- Balance Sheet: Assets, liabilities, equity, total assets
- Cash Flow Statement: Operating, investing, financing activities
- Statement of Retained Earnings: Beginning balance, net income, dividends, ending balance

Always specify which statement contains the requested information."""

FINANCIAL_USER_TEMPLATE = """Question: {question}

Financial Document Content:
{context}

Instructions: Answer the financial question using only the information from the financial statements above. Include specific dollar amounts, dates, and statement references."""


def detect_financial_query_type(query: str) -> str:
    query_lower = query.lower()
    
    # Balance Sheet items
    if any(term in query_lower for term in ['total assets', 'assets', 'liabilities', 'equity', 'balance sheet']):
        return 'balance_sheet'
    
    # Income Statement items  
    elif any(term in query_lower for term in ['revenue', 'income', 'profit', 'expenses', 'sales', 'net income', 'operating income']):
        return 'income_statement'
    
    # Cash Flow items
    elif any(term in query_lower for term in ['cash flow', 'operating activities', 'investing activities', 'financing activities']):
        return 'cash_flow'
    
    # Retained Earnings
    elif any(term in query_lower for term in ['retained earnings', 'dividends', 'retained']):
        return 'retained_earnings'
    
    return 'general_financial'


def extract_financial_amounts(text: str) -> List[Dict[str, Any]]:

    dollar_pattern = r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    amounts = re.findall(dollar_pattern, text)
    
    # Pattern for percentages
    percent_pattern = r'(\d+\.?\d*)\s*%'
    percentages = re.findall(percent_pattern, text)
    
    return {
        'dollar_amounts': amounts,
        'percentages': percentages
    }


def format_financial_context(chunks: List[Dict[str, Any]], query_type: str) -> str:
    if not chunks:
        return "No relevant financial information found."
    
    context_parts = []
    
    # Add header based on query type
    type_headers = {
        'balance_sheet': 'Balance Sheet Information:',
        'income_statement': 'Income Statement Information:',
        'cash_flow': 'Cash Flow Information:',
        'retained_earnings': 'Retained Earnings Information:',
        'general_financial': 'Financial Statement Information:'
    }
    
    context_parts.append(type_headers.get(query_type, 'Financial Information:'))
    context_parts.append("-" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Unknown document")
        page = metadata.get("page_number", "Unknown page")
        
        # Extract financial amounts for highlighting
        content = chunk.get("content", "").strip()
        financial_data = extract_financial_amounts(content)
        
        header = f"[{i}] {source} - Page {page}"
        if financial_data['dollar_amounts']:
            amounts_str = ", ".join(f"${amt}" for amt in financial_data['dollar_amounts'][:3])
            header += f" (Contains: {amounts_str})"
        
        context_parts.append(f"{header}:\n{content}")
    
    return "\n\n".join(context_parts)


def answer_financial_question(question: str, top_k: int = 5, include_debug: bool = False, 
                            target_doc_id: Optional[str] = None) -> Tuple[str, List[Dict], List[str]]:

    debug_history = []
    debug_history.append(f"Financial Question: {question}")
    
    # Detect financial query type
    query_type = detect_financial_query_type(question)
    debug_history.append(f"Detected query type: {query_type}")
    
    # Retrieve with financial context
    chunks = retriever.retrieve(
        question, 
        top_k=top_k, 
        target_doc_id=target_doc_id,
        use_document_context=True
    )
    
    debug_history.append(f"Retrieved {len(chunks)} chunks")
    
    if not chunks:
        answer = "No relevant financial information found in the uploaded documents."
        return answer, [], debug_history
    
    # Format context for financial documents
    context = format_financial_context(chunks, query_type)
    debug_history.append(f"Financial context formatted: {len(context)} characters")
    
    # Use financial-specific prompt
    user_prompt = FINANCIAL_USER_TEMPLATE.format(question=question, context=context)
    
    messages = [
        {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    debug_history.append("Generating financial analysis...")
    
    # Generate response
    response_text = ollama_client.generate(messages, max_tokens=512)
    debug_history.append(f"Financial response generated: {len(response_text)} characters")
    
    # Enhanced provenance for financial data
    provenance_data = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        content = chunk.get("content", "")
        financial_data = extract_financial_amounts(content)
        
        provenance_data.append({
            "doc_id": metadata.get("doc_id", "unknown"),
            "source": metadata.get("source", "unknown"), 
            "page_number": metadata.get("page_number", 0),
            "chunk_type": chunk.get("chunk_type", "text"),
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "relevance_score": chunk.get("final_score", chunk.get("score", 0)),
            "financial_amounts": financial_data['dollar_amounts'],
            "percentages": financial_data['percentages']
        })
    
    return response_text, provenance_data, debug_history


# Update the main answer_question function to use financial enhancement
def answer_question(question: str, top_k: int = 5, include_debug: bool = False, 
                   target_doc_id: Optional[str] = None) -> Tuple[str, List[Dict], List[str]]:

    financial_keywords = ['assets', 'liabilities', 'revenue', 'income', 'profit', 'expenses', 
                         'balance sheet', 'cash flow', '$', 'financial', 'statement']
    
    is_financial = any(keyword in question.lower() for keyword in financial_keywords)
    
    if is_financial:
        return answer_financial_question(question, top_k, include_debug, target_doc_id)
    else:

        debug_history = []
        debug_history.append(f"Question: {question}")
        
        chunks = retriever.retrieve(question, top_k=top_k, target_doc_id=target_doc_id, use_document_context=True)
        debug_history.append(f"Retrieved {len(chunks)} chunks")
        
        if not chunks:
            answer = "No relevant information found in the uploaded documents."
            return answer, [], debug_history
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page_number", "N/A")
            content = chunk.get("content", "").strip()
            context_parts.append(f"[{i}] {source}, Page {page}:\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Basic prompt for non-financial content
        user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer based on the provided context:"
        
        messages = [
            {"role": "system", "content": "Answer questions using only the provided context. Be specific and accurate."},
            {"role": "user", "content": user_prompt}
        ]
        
        response_text = ollama_client.generate(messages, max_tokens=512)
        
        # Basic provenance
        provenance_data = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            provenance_data.append({
                "doc_id": metadata.get("doc_id", "unknown"),
                "source": metadata.get("source", "unknown"),
                "page_number": metadata.get("page_number", 0),
                "chunk_type": chunk.get("chunk_type", "text"),
                "content_preview": chunk.get("content", "")[:200] + "..." if len(chunk.get("content", "")) > 200 else chunk.get("content", ""),
                "relevance_score": chunk.get("final_score", chunk.get("score", 0))
            })
        
        return response_text, provenance_data, debug_history


if __name__ == "__main__":
    # Test financial document query
    test_query = "What are the total assets in $?"
    print(f"Testing: {test_query}")
    
    answer, provenance, debug = answer_question(test_query, include_debug=True)
    
    print(f"Answer: {answer}")
    print(f"Provenance: {len(provenance)} sources")
    if debug:
        print("Debug:", debug[-3:])