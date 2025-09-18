#frontend/app.py

import streamlit as st
import uuid
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import processor, chunker, embedder, qa
from db import utils

print("Starting Streamlit app...")

st.set_page_config(page_title="Financial Doc Q&A", layout="wide")
st.title("📊 Financial Document Q&A")

# Initialize DB
utils.init_db()

# Sidebar upload
st.sidebar.header("Upload Documents")
uploaded = st.sidebar.file_uploader("Upload PDF or Excel", type=["pdf", "xls", "xlsx"])
if uploaded:
    with st.spinner("Processing document..."):
        try:
            doc_json = processor.process_file(uploaded)
            chunks = chunker.chunk_document(doc_json)
            embedder.add_chunks(chunks)
            utils.add_document(doc_json["doc_id"], doc_json["filename"], doc_json["path"])
            st.sidebar.success(f"Processed {doc_json['filename']}")
        except Exception as e:
            st.sidebar.error(f"Error processing document: {str(e)}")

# List available documents
docs = utils.list_documents()
if docs:
    st.sidebar.subheader("Available Documents")
    for d in docs:
        st.sidebar.write(f"📄 {d['filename']} ({d['uploaded_at']})")
else:
    st.sidebar.info("No documents uploaded yet.")

# Chat interface
st.header("Ask a Question")
session_id = st.session_state.get("session_id")
if not session_id:
    session_id = uuid.uuid4().hex
    st.session_state["session_id"] = session_id

query = st.text_input("Enter your question", placeholder="e.g., What are the total assets? What was the net income?")

# Add document targeting option
if docs:
    selected_doc = st.selectbox(
        "Target specific document (optional):",
        ["All documents"] + [d['filename'] for d in docs]
    )
    target_doc_id = None
    if selected_doc != "All documents":
        target_doc_id = next(d['doc_id'] for d in docs if d['filename'] == selected_doc)
else:
    target_doc_id = None

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        try:
            # FIXED: Properly unpack the tuple returned by answer_question
            answer, provenance, debug_history = qa.answer_question(
                query, 
                top_k=5, 
                include_debug=True,
                target_doc_id=target_doc_id
            )
            
            # Store conversation - use the answer string directly
            utils.add_conversation(session_id, query, answer)
            
            # # Display results
            # st.subheader("Answer")
            # st.write(answer)
            
            # Display provenance information
            if provenance:
                st.subheader("Sources")
                for i, prov in enumerate(provenance, 1):
                    with st.expander(f"Source {i}: {prov.get('source', 'filename')} (Page {prov.get('page_number', 'N/A')})"):
                        st.write(f"**Document:** {prov.get('source', 'filename')}")
                        st.write(f"**Page:** {prov.get('page_number', 'N/A')}")
                        st.write(f"**Type:** {prov.get('chunk_type', 'text')}")
                        st.write(f"**Relevance Score:** {prov.get('relevance_score', 0):.3f}")
                        st.write("**Content Preview:**")
                        st.text(prov.get('content_preview', 'No preview available'))
            # Display results
            st.subheader("Answer")
            st.write(answer)
            # st.write("**Content Preview:**")
            st.text(prov.get('content_preview', 'No preview available'))
            
            st.markdown("---")

            # Display debug info if available
            if debug_history and st.checkbox("Show Debug Information"):
                st.subheader("Debug Information")
                for debug_item in debug_history:
                    st.text(debug_item)
                    
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            st.error("Please check if documents are properly uploaded and processed.")

# Test the financial document query
if query and "total assets" in query.lower():
    st.info("💡 For financial documents, I'll look for balance sheet information and specific dollar amounts.")

# Enhanced conversation history with better formatting
if st.checkbox("Show Conversation History"):
    st.header("Conversation History")
    conv = utils.get_conversation(session_id)
    
    if conv:
        for i, turn in enumerate(conv, 1):
            with st.expander(f"Q{i}: {turn['query'][:50]}..." if len(turn['query']) > 50 else f"Q{i}: {turn['query']}"):
                st.markdown(f"**Question:** {turn['query']}")
                st.markdown(f"**Answer:** {turn['response']}")
                st.caption(f"Asked at: {turn['timestamp']}")
    else:
        st.info("No conversation history yet. Ask a question to get started!")

# System info in sidebar
with st.sidebar:
    st.subheader("System Info")
    st.info(f"Session ID: {session_id[:8]}...")
    
    if docs:
        total_size = sum(os.path.getsize(os.path.join("data/uploads", f)) 
                        for f in os.listdir("data/uploads") 
                        if os.path.isfile(os.path.join("data/uploads", f)))
        st.info(f"Total documents: {len(docs)}")
        st.info(f"Storage used: {total_size/1024/1024:.1f} MB")
    
    # Quick test buttons
    st.subheader("Quick Tests")
    if st.button("Test: What are the total assets?"):
        st.rerun()
    
    if st.button("Test: What was the net income?"):
        st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 **Tips:** Ask specific questions about financial data, upload multiple documents for comparison, use the document selector to target specific files.")
# Conversation history
# st.header("Conversation History")
# conv = utils.get_conversation(session_id)
# for turn in conv:
#     st.markdown(f"**Q:** {turn['query']}")
#     st.markdown(f"**A:** {turn['response']}")
#     st.caption(turn['timestamp'])
