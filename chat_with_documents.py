import streamlit as st
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv, find_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma


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
def chunk_data(data, chunk_size=256, chunk_overlap=20):
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
def create_embeddings_chroma(chunks, embeddings, persist_directory="./chroma_db"):
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

def ask_and_get_answer(query, vector_store, llm, k=3):
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
    
if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)

    llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    st.image("image.png")
    st.subheader("LLM Q&A with Documents App")

    with st.sidebar:
        # Modify the snippet below to set your OpenAI/Anthropic/Mistral API key
        #api_key = st.text_input("Enter your GROQ API key", type="password")
        #if api_key:
        #    os.environ["GROQ_API_KEY"] = api_key
        #    st.success("API key set successfully!")
        #else:
        #    st.warning("Please enter your GROQ API key.")
        
    
        uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])
        chunk_size = st.number_input("Chunk size:", min_value=100, max_value=2048, value=200)
        k = st.number_input("k", min_value=1, max_value=10, value=3)
        add_data = st.button("Add Data")

        if uploaded_file and add_data:
            with st.spinner("Reading, Chunking and embedding file..."):
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f"Chunk Size:{chunk_size}, Number of chunks: {len(chunks)}")

                vector_store = create_embeddings_chroma(chunks, embeddings)

                st.session_state.vector_store = vector_store
                st.success("File uploaded, chunked and embedded successfully!")

    q= st.text_input("Ask a question about the document:")
    if q and 'vector_store' in st.session_state:
         with st.spinner("Getting answer..."):
            st.write(f"K: {k}")
            answer = ask_and_get_answer(q, st.session_state.vector_store, llm, k=k)
            st.text_area("LLM Answer:", value=answer['result'])
