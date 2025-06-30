import streamlit as st
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv, find_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

load_dotenv(find_dotenv(), override=True)


llm = ChatGroq(model="llama3-8b-8192", temperature=1)

response = llm.invoke("What is the capital of Finland?")
print(response.content)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

## Load document
def load_document(file_path):
    """
    Load a document from the specified file path.
    Supports PDF, DOCX, and TXT formats.
    Args:
        file_path (str): The path to the document file.
        Returns:documents"""
    name, extension = os.path.splitext(file_path)
    
    if extension.lower() == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading document {file_path}...")
        loader = PyPDFLoader(file_path)
    elif extension.lower() == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading document {file_path}...")
        loader = Docx2txtLoader(file_path)
    elif extension.lower() == '.txt':
        from langchain.document_loaders import TextLoader
        print(f"Loading document {file_path}...")
        loader = TextLoader(file_path)
    else:
        return None
    
        data = loader.load()
        return data
    
    ## Chunking the data
    def chunk(data, chunk_size=256, chunk_overlap=20):
        """
        Chunk the data into smaller pieces.
        Args:
            data (list): List of documents to chunk.
        Returns: List of chunks.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(data)
        return chunks

    ## Create embeddings and store them in ChromaDB
    def create_embeddings_chroma(chunks, persist_directory="./chroma_db"):
        """
        Create embeddings and store them in ChromaDB.
        Args:
            chunks (list): List of document chunks to embed and store.
            persist_directory (str): Directory to persist the ChromaDB.
        Returns: ChromaDB vector store.
        """
        from langchain_community.vectorstores import Chroma

        vector_store = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_directory)
        #vector_store.persist()

        return vector_store

    
    def ask_and_get_answer(query, vector_store,k=3):
        """
        Ask a question and get an answer from the vector store.
        Args:
            query (str): The question to ask.
            vector_store (Chroma): The ChromaDB vector store.
        Returns: Answer to the query.
        """
        from langchain.chains import RetrievalQA

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        answer = qa_chain.invoke(query)

        return answer