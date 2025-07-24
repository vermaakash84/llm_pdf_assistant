
# 🤖 LLM PDF Research Assistant App (LangChain + FAISS)

Ask questions from your legal or research PDFs and get AI-powered answers with source citations — right in your browser using Streamlit!

---

## 🔍 Features

- ✅ Ask natural language questions to your PDF documents.
- ✅ Supports **batch switching** between different vector stores (`faiss_batch_1`, `faiss_batch_2`).
- ✅ Displays relevant **source documents** used for each answer.
- ✅ Powered by `LangChain`, `FAISS`, and Hugging Face models.
- ✅ Deployed with Streamlit Cloud.

---

## 📁 Folder Structure

```
llm-pdf-assistant-app/
├── app.py                      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── faiss_batch_1/              # Vector store for Batch 1 PDFs
│   ├── index.faiss
│   └── index.pkl
├── faiss_batch_2/              # Vector store for Batch 2 PDFs
│   ├── index.faiss
│   └── index.pkl
└── README.md
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/vermaakash84/llm-pdf-assistant-app.git
cd llm-pdf-assistant-app

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🌐 Live Demo

> 🔗 [Streamlit App](https://llm-pdf-assistant-app-akash.streamlit.app)  
> 🔗 [GitHub Repo](https://github.com/vermaakash84/llm-pdf-assistant-app)

---

## 🧠 Powered By

- [LangChain](https://www.langchain.com/)
- [FAISS Vector Store](https://github.com/facebookresearch/faiss)
- [Transformers (Flan-T5)](https://huggingface.co/google/flan-t5-base)
- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## ✍️ Author

Built with ❤️ by [Akash Verma](https://www.linkedin.com/in/vermaakash84)
