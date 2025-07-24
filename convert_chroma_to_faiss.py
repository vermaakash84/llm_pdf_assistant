from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Customize the batch you want to convert
chroma_dir = "db_batch_2"
faiss_dir = "faiss_batch_2"

# Load the same embedding model you use in your app
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Chroma vector store
chroma_db = Chroma(persist_directory=chroma_dir, embedding_function=embedding)

# Get all documents stored in Chroma
docs = chroma_db.get().documents
metas = chroma_db.get().metadatas

# Rebuild FAISS from documents and metadata
faiss_db = FAISS.from_texts(texts=docs, embedding=embedding, metadatas=metas)

# Save FAISS index
faiss_db.save_local(faiss_dir)

print(f" Converted {chroma_dir} → {faiss_dir}")
