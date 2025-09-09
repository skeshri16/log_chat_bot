import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def get_embeddings():
    """Load HuggingFace embeddings only once and cache them in Streamlit session."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(chunks):
    """Create FAISS vectorstore from chunks using cached embeddings."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore
