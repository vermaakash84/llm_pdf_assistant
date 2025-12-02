# ---------------------------------------------------
# Fix for Chroma DB SQLite version issue - MUST be first
# ---------------------------------------------------
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# ---------------------------------------------------
# STREAMLIT CONFIG — Fast loading, wide layout
# ---------------------------------------------------
st.set_page_config(page_title="⚡ Ultra-Fast PDF Assistant", layout="wide")
st.title("⚡ Ultra-Fast LLM PDF Research Assistant")


# ---------------------------------------------------
# Sidebar — Batch selector
# ---------------------------------------------------
batch = st.sidebar.selectbox("📁 Chroma DB Batch", ["db_batch_1", "db_batch_2"])
extract_path = f"{batch}_chroma"
st.sidebar.success(f"Using vector DB: `{extract_path}`")


# ---------------------------------------------------
# Load Chroma Vector DB (Fast Index)
# ---------------------------------------------------
if not os.path.exists(extract_path):
    st.error(f"❌ Chroma DB folder not found: `{extract_path}`")
    st.stop()

try:
    st.sidebar.info("🔄 Loading Embeddings…")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=extract_path,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}  # FASTEST search
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})  # fewer docs = faster
    st.sidebar.success("⚡ Vector DB Ready!")
except Exception as e:
    st.error(f"Vector DB Error: {e}")
    st.stop()


# ---------------------------------------------------
# FAST LLM: FLAN-T5 in 8-bit (no meta tensor errors)
# ---------------------------------------------------
st.sidebar.info("🧠 Loading FLAN-T5 (8-bit optimized)…")

try:
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load with 8-bit quantization → MUCH faster + low RAM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",        # puts model on correct device
        load_in_8bit=True,        # huge speed boost
        torch_dtype=torch.float16
    )

    # Fast generation pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        repetition_penalty=1.1,
        device_map="auto",
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    st.sidebar.success("🚀 Model Ready (8-bit Fast Mode)")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ---------------------------------------------------
# RAG Chain (LCEL)
# ---------------------------------------------------
prompt = PromptTemplate.from_template(
    """You are a helpful research assistant. 
Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)

rag_chain = (
    RunnableParallel(
        context=(
            retriever 
            | (lambda docs: "\n\n".join(d.page_content for d in docs))
        ),
        question=RunnablePassthrough()
    )
    | prompt
    | llm
)


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.markdown("💬 Ask a question based on the embedded PDF knowledge base.")

query = st.text_input("🔍 Enter your question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🤖 Thinking…"):
            try:
                result = rag_chain.invoke(query)
                st.subheader("🧠 Answer")
                st.write(result)
            except Exception as e:
                st.error(f"❌ Failed: {e}")
