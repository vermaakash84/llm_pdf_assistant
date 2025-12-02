# -----------------------------
# Fix for Chroma DB SQLite issue
# -----------------------------
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import torch

# Chroma + LangChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# HuggingFace Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

st.set_page_config(page_title="📄 LLaMA 3.2 RAG PDF Assistant")

# ---------------------------------------------------
# Sidebar – Batch Selector
# ---------------------------------------------------
batch = st.sidebar.selectbox("📁 Select Chroma DB Batch", ["db_batch_1", "db_batch_2"])
extract_path = f"{batch}_chroma"

# ---------------------------------------------------
# Load Chroma Vector DB
# ---------------------------------------------------
st.sidebar.success(f"Loading Chroma DB: `{extract_path}`")

if not os.path.exists(extract_path):
    st.error(f"❌ Database not found: {extract_path}")
    st.stop()

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=extract_path, embedding_function=embeddings)
    retriever = db.as_retriever()
except Exception as e:
    st.error(f"❌ Error loading Chroma DB: {e}")
    st.stop()

# ---------------------------------------------------
# Load LLaMA 3.2 (FREE, Local CPU)
# ---------------------------------------------------
st.sidebar.info("🦙 Loading LLaMA 3.2 (FREE, local CPU)... Please wait (10–20 sec)...")

try:
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,   # CPU friendly
        device_map="cpu"             # FORCE CPU
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        device=-1                   # FORCE CPU INFERENCE
    )

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ---------------------------------------------------
# RAG Prompt
# ---------------------------------------------------
prompt = PromptTemplate.from_template("""
You are an accurate research assistant.

Use ONLY the following context to answer the question.
If the answer is not found, say "I cannot find this information in the documents."

Context:
{context}

Question:
{question}

Answer:
""")

# ---------------------------------------------------
# LCEL RAG Chain
# ---------------------------------------------------
def join_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    RunnableParallel(
        context=retriever | join_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | (lambda x: pipe(x, max_new_tokens=300)[0]["generated_text"])
)

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("📄 LLaMA 3.2 – FREE PDF Research Assistant")
st.markdown("Ask questions based on your PDF embeddings (Fully Free & Local).")

query = st.text_input("🔍 Enter your question:")

if st.button("🚀 Get Answer") and query:
    with st.spinner("Thinking..."):
        try:
            result = rag_chain.invoke(query)
            st.subheader("🧠 Answer:")
            st.write(result)
        except Exception as e:
            st.error(f"❌ Error: {e}")
