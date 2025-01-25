from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_FAISS_PATH = 'vectorstore\db_faiss'

def create_embeddings():

    loader = PyPDFDirectoryLoader("Data")
    text = loader.load()

    text_splitter = CharacterTextSplitter(separator='Question:', chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(text)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    embeddings_hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings_hf,
    )
    
    vectorstore.save_local(DB_FAISS_PATH)
    print("Succesfully created embedding")

if __name__ == "__main__":
    create_embeddings()
