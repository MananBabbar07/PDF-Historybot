# GenAI PDF Question Answering System (RAG)

## Overview
This project is a **Retrieval-Augmented Generation (RAG)** based GenAI application that enables users to upload a PDF document and ask questions that are answered **strictly from the document’s content**.  
The system is intentionally designed to **avoid hallucinations** and responds with *“I don’t know”* when the answer is not present in the PDF.

This project demonstrates practical GenAI engineering concepts such as document ingestion, semantic retrieval, prompt control, and conversational memory.

---

## Key Features
- PDF upload and processing
- Document-grounded question answering (RAG)
- Semantic search using embeddings and vector database
- Conversational chat history
- Hallucination control (no out-of-context answers)
- Simple Streamlit web interface

---

## Tech Stack
- **Python**
- **Streamlit** – UI
- **LangChain** – RAG pipeline
- **Groq LLM (LLaMA-3.1-8B)** – Language model
- **ChromaDB** – Vector database
- **HuggingFace Sentence Transformers** – Embeddings
- **PyPDFLoader / Unstructured PDF Loader** – PDF text extraction

---

## How the System Works
1. A PDF document is uploaded by the user.
2. The document is split into chunks and converted into vector embeddings.
3. Embeddings are stored in a Chroma vector database.
4. User queries are matched with relevant document chunks.
5. The LLM generates answers using **only retrieved context**.
6. If the information is not found in the PDF, the system explicitly responds with *“I don’t know”*.

---
## Installation & Usage

Clone the repository and move into the project directory:
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

Create and activate a virtual environment:
python -m venv venv
Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate

Install required dependencies:
pip install -r requirements.txt

Create a `.env` file in the project root and add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here

Run the application:
streamlit run app.py

Using the application:
- Enter your Groq API key in the UI
- Upload a PDF document
- Ask questions related to the document
- The system answers strictly based on the PDF content and responds with “I don’t know” if information is not present
