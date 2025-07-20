import streamlit as st
import pandas as pd
import os
import tempfile
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import matplotlib.pyplot as plt

# Set API key from environment
genai.configure(api_key="AIzaSyBEayIJuF0HVrqvZ-W5t6VDdHinlnhy_Nk")

# Set app title
st.set_page_config(page_title="📊 RAG AI Assistant with Gemini")
st.title("📊 CSV RAG Assistant with Gemini + ChromaDB")

@st.cache_data
def load_and_prepare_chunks(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    chunks = []
    for i, row in df.iterrows():
        row_text = " | ".join([f"{col} = {row[col]}" for col in df.columns])
        chunks.append(Document(page_content=row_text))
    return chunks, df

@st.cache_resource
def create_vectorstore(_chunks, persist_path):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(chunks)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_path)
    vectorstore.persist()
    return vectorstore

def ask_question_with_gemini(query, vectorstore, model, top_k=5):
    results = vectorstore.similarity_search(query, k=top_k)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"""
You are a helpful assistant analyzing a CSV dataset.

Use the following context to answer:
{context}

Question:
{query}

Answer:
"""
    response = model.generate_content(prompt)
    return response.text

# Upload section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    chunks, df = load_and_prepare_chunks(temp_path)
    vectorstore = create_vectorstore(chunks, persist_path="./rag_db")
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    st.subheader("🤖 Ask a Question")
    user_question = st.text_input("Type your question (e.g., 'Total sales in 2023')")

    if user_question:
        with st.spinner("Thinking..."):
            answer = ask_question_with_gemini(user_question, vectorstore, gemini_model)
        st.markdown("### ✅ Gemini’s Answer")
        st.write(answer)

    # Optional: Visualization
    st.subheader("📈 Visualization (Optional)")
    if st.button("Show column distribution"):
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            st.selectbox("Select a column", numeric_cols, key="selected_col")
            selected_col = st.session_state.selected_col

            fig, ax = plt.subplots()
            df[selected_col].hist(ax=ax, bins=20)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for plotting.")

else:
    st.info("👆 Upload a CSV file to begin")
