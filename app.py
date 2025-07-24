import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# ---------------------- SETUP ----------------------

st.set_page_config(page_title="LLM PDF Assistant", layout="wide")
st.title("📚 LLM PDF Research Assistant")

# Dropdown to select batch
batch_option = st.selectbox(
    "Select vector store batch:",
    ("faiss_batch_1", "faiss_batch_2")
)

# Load embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector store
@st.cache_resource
def load_vectorstore(batch_name):
    embeddings = load_embeddings()
    return FAISS.load_local(
        folder_path=batch_name,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

# Load QA chain
@st.cache_resource
def load_qa_chain(vectordb):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# ---------------------- UI ----------------------

st.info(f"🔍 Using vector store: `{batch_option}`")

# Load selected FAISS vector store
vectordb = load_vectorstore(batch_option)
qa_chain = load_qa_chain(vectordb)

# User query input
query = st.text_input("💬 Ask a question based on the PDFs:", "")

# Run QA on query
if query:
    with st.spinner("🔎 Generating answer..."):
        response = qa_chain.invoke({"query": query})
        st.subheader("🧠 Answer")
        st.write(response["result"])

        st.subheader("📚 Source Documents")
        for doc in response["source_documents"]:
            st.markdown(f"- `{doc.metadata['source']}`")
