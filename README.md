
# 📄 LLM PDF Research Assistant (LangChain + Chroma + Streamlit)

Ask intelligent questions from 400+ PDF documents using a local Large Language Model (LLM), vector search, and a beautiful Streamlit interface. Supports **batch toggle** with pre-built ChromaDB vector stores stored as ZIPs for efficient cloud deployment.

![LLM PDF Assistant Banner](https://i.imgur.com/INvPIVW.png)

---

## 🚀 Features

- 🔍 **Semantic search** over PDFs using `ChromaDB` and `MiniLM` embeddings  
- 🤖 **Question-answering** with `FLAN-T5` from Hugging Face  
- 📂 **Batch switching** to load different document sets  
- ⚡ **Fast & lightweight** — no OpenAI API key needed  
-☁️ **Streamlit Cloud-ready** with ZIP-compressed vector DBs  

---

## 🧠 Tech Stack

| Tool/Library       | Purpose                             |
|--------------------|-------------------------------------|
| `LangChain`        | RetrievalQA and LLM orchestration   |
| `ChromaDB`         | Vector store for document chunks    |
| `Transformers`     | FLAN-T5 model for QA                |
| `HuggingFace`      | Embeddings & model loading          |
| `Streamlit`        | UI and deployment                   |
| `zipfile`, `os`    | Dynamic unzip on load               |

---

## 📁 Folder Structure

```
llm_pdf_assistant/
├── app.py                     # Main Streamlit app
├── db_batch_1.zip             # Chroma ZIP for batch 1
├── db_batch_2.zip             # Chroma ZIP for batch 2
├── requirements.txt           # Streamlit Cloud requirements
├── .gitignore
└── README.md
```

---

## ✅ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/vermaakash84/llm_pdf_assistant.git
   cd llm_pdf_assistant
   ```

2. **Create a virtual environment (optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deploy to Streamlit Cloud

Just push your code (with `app.py`, `requirements.txt`, and zipped `db_batch_*.zip` files) to your GitHub repo and deploy on [streamlit.io/cloud](https://streamlit.io/cloud). It auto-detects the app!

---

## 📷 Preview

![App Screenshot](https://i.imgur.com/XqFgYJP.png)

---

## 🙋‍♂️ About Me

Made with ❤️ by [Akash Verma](https://www.linkedin.com/in/vermaakash84)
