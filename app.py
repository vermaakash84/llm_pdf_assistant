# ================================================================
# Fix for Chroma DB SQLite version issue - MUST be at the very top
# ================================================================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import os
import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="📄 LLM PDF Research Assistant")

# ==========================================================
# SIDEBAR → Batch selection
# ==========================================================
batch = st.sidebar.selectbox("📁 Select Chroma DB Batch", ["db_batch_1", "db_batch_2"])
extract_path = f"{batch}_chroma"

st.sidebar.success(f"Loading Chroma vector DB: `{extract_path}`")

# ==========================================================
# STEP 1 — Load Chroma DB
# ==========================================================
if not os.path.exists(extract_path):
    st.error(f"❌ Chroma DB folder not found: `{extract_path}`")
    st.stop()

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory=extract_path,
        embedding_function=embeddings
    )

    retriever = db.as_retriever()

except Exception as e:
    st.error(f"❌ Failed to load Chroma DB: {e}")
    st.stop()


# ==========================================================
# STEP 2 — Load FLAN-T5 model (CPU Optimized)
# ==========================================================
st.sidebar.info("🔄 Loading FLAN-T5 model (CPU optimized)...")

try:
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map=None,               # Disable GPU mapping
        torch_dtype=torch.float32      # Force CPU dtype
    ).to("cpu")                        # Ensure model on CPU

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        device=-1                      # Force CPU inference
    )

    llm = HuggingFacePipeline(pipeline=pipe)

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()


# ==========================================================
# STEP 3 — Build LCEL RAG Chain
# ==========================================================
def join_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

prompt = PromptTemplate.from_template(
"""
You are a helpful research assistant. Use ONLY the following context to answer:

{context}

Question: {question}

Answer:
"""
)

rag_chain = (
    RunnableParallel(
        context=retriever | join_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
)


# ==========================================================
# STEP 4 — Streamlit UI
# ==========================================================
st.title("📄 LLM PDF Research Assistant")
st.markdown("Ask a question based on the 400+ PDF document embeddings.")

query = st.text_input("🔍 Ask your question:")

if st.button("🚀 Get Answer") and query.strip():
    with st.spinner("🤖 Thinking..."):
        try:
            response = rag_chain.invoke(query)
            st.subheader("🧠 Answer")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Failed to answer your query: {e}")
