import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os

st.title("📚 LLM PDF Research Assistant")

# Select batch
batch_choice = st.selectbox("Select a document batch", ["faiss_batch_1", "faiss_batch_2"])
db_path = batch_choice

# Load embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vector store
try:
    vectordb = FAISS.load_local(db_path, embedding_function=embedding_model)
except Exception as e:
    st.error(f"Vector store could not be loaded from: {db_path}")
    st.stop()

# Load Hugging Face LLM
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Setup RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Query input
query = st.text_input("Enter your question:")
if query:
    result = qa_chain.invoke({"query": query})
    st.subheader("🧠 Answer")
    st.write(result["result"])
    st.subheader("📄 Source Documents")
    for doc in result["source_documents"]:
        st.markdown(f"- `{doc.metadata.get('source', 'Unknown')}`")
