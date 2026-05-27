# Simple PDF Chatbot using RAG
# ---------------------------------
# Step 1: Upload PDF
# Step 2: Read PDF text
# Step 3: Split text into chunks
# Step 4: Create embeddings
# Step 5: Store embeddings in FAISS
# Step 6: Ask questions from PDF

import streamlit as st
from pypdf import PdfReader

# LangChain Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# -------------------------------
# OpenAI API Key
# -------------------------------
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📄 Simple PDF Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# -------------------------------
# Process PDF
# -------------------------------
if uploaded_file is not None:

    # Read PDF
    pdf = PdfReader(uploaded_file)

    # Extract text from all pages
    text = ""

    for page in pdf.pages:
        text += page.extract_text()

    # Show extracted text size
    st.write("PDF Loaded Successfully")

    # -------------------------------
    # Split Text into Chunks
    # -------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    st.write(f"Total Chunks Created: {len(chunks)}")

    # -------------------------------
    # Create Embeddings
    # -------------------------------
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    # -------------------------------
    # Store in FAISS Vector DB
    # -------------------------------
    vector_store = FAISS.from_texts(chunks, embeddings)

    # -------------------------------
    # User Question
    # -------------------------------
    question = st.text_input("Ask Question From PDF")

    # -------------------------------
    # Search + Generate Answer
    # -------------------------------
    if question:

        # Convert vector DB into retriever
        retriever = vector_store.as_retriever()

        # Find similar chunks
        docs = retriever.get_relevant_documents(question)

        # Combine retrieved chunks
        context = ""

        for doc in docs:
            context += doc.page_content

        # LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            temperature=0
        )

        # Final Prompt
        prompt = f"""
        Answer the question using the below context.

        Context:
        {context}

        Question:
        {question}
        """

        # Generate Answer
        response = llm.invoke(prompt)

        # Show Answer
        st.subheader("Answer")
        st.write(response.content)
