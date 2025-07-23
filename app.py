import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

st.title("📚 LLM PDF Research Assistant")

# Select batch
batch_choice = st.selectbox("Select a document batch", ["rag_batch_1", "rag_batch_2"])
db_path = f"db_batch_{batch_choice.split('_')[-1]}"

# Load vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=db_path, embedding_function=embedding_model)

# Load HF model
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Setup RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# User query
query = st.text_input("Enter your question:")
if query:
    result = qa_chain.invoke({"query": query})
    st.subheader("🧠 Answer")
    st.write(result["result"])
    st.subheader("📄 Source Documents")
    for doc in result["source_documents"]:
        st.markdown(f"- `{doc.metadata['source']}`")
