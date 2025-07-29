# Fix for Chroma DB SQLite version issue - MUST be at the very top
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="📄 LLM PDF Research Assistant")

# -----------------------------
# Sidebar: Batch toggle
# -----------------------------
batch = st.sidebar.selectbox("📁 Select Chroma DB Batch", ["db_batch_1", "db_batch_2"])
extract_path = f"{batch}_chroma"

# -----------------------------
# Step 1: Load Chroma vector DB
# -----------------------------
st.sidebar.success(f"Loading Chroma vector DB: `{extract_path}`")
if not os.path.exists(extract_path):
    st.error(f"❌ Chroma DB folder not found: `{extract_path}`")
    st.stop()

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=extract_path, embedding_function=embeddings)
    retriever = db.as_retriever()
except Exception as e:
    st.error(f"Error loading Chroma DB: {e}")
    st.stop()

# -----------------------------
# Step 2: Load FLAN-T5 model
# -----------------------------
st.sidebar.info("🔄 Loading FLAN-T5 model...")
try:
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )
    llm = HuggingFacePipeline(pipeline=pipe)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -----------------------------
# Step 3: RetrievalQA chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -----------------------------
# Step 4: Streamlit UI
# -----------------------------
st.title("📄 LLM PDF Research Assistant")
st.markdown("Ask a question based on the 400+ PDF document embeddings.")

query = st.text_input("🔍 Ask your question:")
if st.button("🚀 Get Answer") and query:
    with st.spinner("🤖 Thinking..."):
        try:
            response = qa_chain.invoke(query)
            st.subheader("🧠 Answer")
            st.write(response['result'])

            st.subheader("📄 Source Documents")
            for doc in response['source_documents'][:3]:
                st.markdown(f"- `{doc.metadata.get('source', 'Unknown')}`")
        except Exception as e:
            st.error(f"❌ Failed to answer: {e}")
