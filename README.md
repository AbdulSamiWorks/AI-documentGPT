# 📄 DocumentGPT – Chat with Your Files using LLMs

DocumentGPT is a **Streamlit** application that allows users to upload documents (PDF, DOCX), process them into vector embeddings using **Hugging Face models**, and then ask questions about the content using **OpenAI's GPT-3.5-Turbo** with retrieval-augmented generation (RAG).

---

## 🧠 Features

- 📁 Upload and process multiple PDF files
- ✂️ Automatic text chunking
- 🧠 Embedding via `all-MiniLM-L6-v2` (HuggingFace)
- 🔎 FAISS vector store for semantic search
- 💬 Conversational retrieval using GPT-3.5
- 🧵 Persistent chat memory across turns
- ⚡ Streamlit chat interface

---

## 🚀 How It Works

1. **Upload Documents** via the sidebar.
2. **Text Extraction** happens for PDFs and DOCX files.
3. **Chunking** splits long documents into manageable pieces.
4. **Vectorization** uses HuggingFace embeddings (`sentence-transformers`).
5. **FAISS** stores these chunks as vectors.
6. **Conversational Chain** retrieves relevant info based on user queries.
7. **LLM** (OpenAI GPT-3.5) generates answers in a conversational flow.

---

## 🛠️ Tech Stack

- `Streamlit` – frontend
- `PyPDF2`, `python-docx` – file parsing
- `LangChain` – orchestration of LLM + retrieval
- `FAISS` – vector store
- `HuggingFace Embeddings` – text embedding
- `OpenAI GPT-3.5 Turbo` – conversational LLM
- `dotenv` – API key handling
- `streamlit-chat` – for cleaner chat UI

---

---

## 🔐 Environment Setup

Create a `.env` file in the root:

```
OPENAI_API_KEY=your_openai_key_here
```

Or, for Streamlit deployment, store it in `.streamlit/secrets.toml`:

```toml
[general]
OPENAI_API_KEY = "your_openai_key_here"
```

---

## ✅ Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/documentgpt
cd documentgpt

# Step 2: Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
streamlit run app.py
```

---

## 📝 To Do

* [ ] Add DOCX and CSV support
* [ ] Add support for larger document uploads (LangChain text splitting improvements)
* [ ] Deploy on Streamlit Cloud or HuggingFace Spaces

---

## 📸 Demo Preview

> *You can include screenshots or a GIF here for better visualization.*

---

## 🤝 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [OpenAI](https://openai.com/)
* [Sentence Transformers](https://www.sbert.net/)
* [Streamlit](https://streamlit.io/)
* [FAISS](https://github.com/facebookresearch/faiss)

---

## 🧑‍💻 Author

Made with ❤️ by [Abdul Sami](https://github.com/AbdulSamiWorks)

---


