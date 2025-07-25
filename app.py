import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Sidebar: Batch toggle
batch = st.sidebar.selectbox("Select Batch", ["db_batch_1", "db_batch_2"])

# Load vector DB
path = f"./{batch}_chroma"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=path, embedding_function=embeddings)
retriever = db.as_retriever()

# Load model pipeline
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Streamlit UI
st.title("📄 LLM PDF Research Assistant")
query = st.text_input("Ask your question:")
if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(query)
        st.subheader("🧠 Answer")
        st.write(response['result'])

        st.subheader("📄 Source Documents")
        for doc in response['source_documents'][:3]:
            st.markdown(f"- {doc.metadata['source']}")
