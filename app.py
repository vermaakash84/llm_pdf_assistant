import streamlit as st
import os
import zipfile
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ updated import
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------------
# Sidebar: Batch toggle
# -----------------------------
batch = st.sidebar.selectbox("Select Batch", ["db_batch_1", "db_batch_2"])

# -----------------------------
# Step 1: Unzip if not already
# -----------------------------
zip_path = f"{batch}.zip"
extract_path = f"{batch}_chroma"

if not os.path.exists(extract_path):
    st.sidebar.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# -----------------------------
# Step 2: Load Chroma vector DB
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=extract_path, embedding_function=embeddings)
retriever = db.as_retriever()

# -----------------------------
# Step 3: Load FLAN-T5 model
# -----------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# Step 4: RetrievalQA chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# -----------------------------
# Step 5: Streamlit UI
# -----------------------------
st.title("📄 LLM PDF Research Assistant")

query = st.text_input("Ask your question:")
if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(query)

        st.subheader("🧠 Answer")
        st.write(response['result'])

        st.subheader("📄 Source Documents")
        for doc in response['source_documents'][:3]:
            st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")