import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# -------------------- Streamlit UI --------------------
st.title("Conversational RAG with PDF + Chat History")
st.write("Upload a PDF and ask questions grounded in its content.")

api_key = st.text_input("Enter your Groq API Key", type="password")

# -------------------- Session State Init --------------------
if "store" not in st.session_state:
    st.session_state.store = {}

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------- Embeddings --------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------- LLM --------------------
if not api_key:
    st.warning("Please enter your Groq API Key")
    st.stop()

llm = ChatGroq(
    groq_api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0
)

session_id = st.text_input("Session ID", value="default_session")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# -------------------- PDF Upload --------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF..."):
        # Save temp PDF
        temp_pdf = "temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load & split
        loader = PyPDFLoader(temp_pdf)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        st.write(f"PDF chunks loaded: {len(splits)}")

        # Vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        # -------------------- Prompts --------------------
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question, "
             "rephrase the question so it is standalone."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using ONLY the provided context. "
             "If the answer is not in the context, say you don't know.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.rag_chain = conversational_rag_chain

        st.success("PDF processed successfully. You can now ask questions!")

# -------------------- Chat --------------------
if st.session_state.rag_chain:
    user_input = st.text_input("Ask a question about the PDF")
    if user_input:
        response = st.session_state.rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.write("### Assistant")
        st.write(response["answer"])

        st.write("### Chat History")
        st.write(get_session_history(session_id).messages)
else:
    st.info("Upload a PDF to start asking questions.")
