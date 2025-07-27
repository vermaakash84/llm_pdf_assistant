
# 📄 LLM PDF Research Assistant (LangChain + ChromaDB)

Ask questions from research/legal PDFs using a fine-tuned FLAN-T5 model and vector search powered by ChromaDB.

## 🚀 Deploy with One Click on Streamlit Cloud

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

> **Instructions:**
> - Make sure your repo is public or linked with Streamlit Cloud.
> - Upload the zipped vector stores: `db_batch_1.zip` and `db_batch_2.zip`.
> - Then click the button above to deploy.

## 🛠 Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Folder Structure

```
.
├── app.py
├── db_batch_1.zip
├── db_batch_2.zip
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔁 Features

- Batch toggle: `db_batch_1` or `db_batch_2`
- FLAN-T5 for answering questions
- Source document reference

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Hugging Face Transformers](https://huggingface.co)
- [Streamlit](https://streamlit.io)
- [ChromaDB](https://www.trychroma.com)
