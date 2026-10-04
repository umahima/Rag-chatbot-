import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# ------------------------------------
# LOAD ENV
# ------------------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ------------------------------------
# PDF → TEXT
# ------------------------------------
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

# ------------------------------------
# TEXT → CHUNKS
# ------------------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# ------------------------------------
# CHUNKS → VECTOR STORE
# ------------------------------------
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)

# ------------------------------------
# RAG CHAIN (Retriever → Context → Gemini)
# ------------------------------------
def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant.
        Answer the question ONLY using the provided context.
        If the answer is not in the context, say:
        "The answer is not available in the provided PDF."

        Context:
        {context}

        Question:
        {input}
        """
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return rag_chain

# ------------------------------------
# STREAMLIT UI
# ------------------------------------
def main():
    st.set_page_config("RAG PDF Chatbot", page_icon="📄")
    st.header("📄 RAG Chatbot using Gemini 2.5 Flash")

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    question = st.text_input("Ask a question from the PDF")

    if question and st.session_state.rag_chain:
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(
                {"input": question}
            )
            st.subheader("✅ Answer")
            st.write(response["answer"])

    with st.sidebar:
        st.subheader("Upload PDF")
        pdf_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDF"):
            if not pdf_files:
                st.warning("Please upload at least one PDF")
            else:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_files)
                    chunks = split_text(raw_text)
                    vectorstore = create_vectorstore(chunks)
                    st.session_state.rag_chain = create_rag_chain(vectorstore)
                    st.success("✅ PDF processed successfully!")

if __name__ == "__main__":
    main()




