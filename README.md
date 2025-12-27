# GenAI PDF Question Answering System (RAG)

### 🚀 Live Demo
https://pdf-historybot.streamlit.app

## Overview
This project is a **Retrieval-Augmented Generation (RAG)** based GenAI application that allows users to upload a PDF document and ask questions that are answered **strictly from the document’s content**.  
The system is intentionally designed to **avoid hallucinations** and responds with **“I don’t know”** when the required information is not present in the PDF.

The project demonstrates practical GenAI engineering concepts such as document ingestion, semantic retrieval, prompt control, and conversational memory.

---

## Key Features
- PDF upload and processing
- Document-grounded question answering (RAG)
- Semantic search using embeddings and a vector database
- Conversational chat history support
- Hallucination control (no out-of-context answers)
- Simple and interactive Streamlit web interface

---

## Tech Stack
- **Python**
- **Streamlit** – Web UI
- **LangChain** – RAG pipeline orchestration
- **Groq LLM (LLaMA-3.1-8B)** – Language model
- **ChromaDB** – Vector database
- **HuggingFace Sentence Transformers** – Embeddings
- **PyPDFLoader** – PDF text extraction

---

## How the System Works
1. A user uploads a PDF document.
2. The document is split into chunks and converted into vector embeddings.
3. Embeddings are stored in a Chroma vector database.
4. User queries are matched with relevant document chunks.
5. The LLM generates answers using **only the retrieved context**.
6. If the answer is not found in the document, the system explicitly responds with **“I don’t know.”**

---

## Live Usage
1. Open the live demo link.
2. Enter your Groq API key.
3. Upload a PDF document.
4. Ask questions related to the document.
5. Receive answers grounded strictly in the PDF content.

---

## Local Setup (Optional)

Clone the repository:
