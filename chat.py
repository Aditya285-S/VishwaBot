from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.environ.get('GROQ_API')
LANGSMITH_API = os.environ.get('LANGSMITH_API')

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = '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'
os.environ["LANGSMITH_PROJECT"] = "Vishwabot"

DB_FAISS_PATH = 'vectorstore\db_faiss'

def get_response(text):

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    embeddings_hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings_hf, allow_dangerous_deserialization=True)

    llm = ChatGroq(
        temperature=0.5, 
        model_name="mixtral-8x7b-32768", 
        groq_api_key='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'
    )


    rag_template = """
        You are a helpful assistant for a college website.

        1. If the user's question is related to specific documents, use the provided context to answer the question and include relevant details from it.
        2. If the user's query is conversational (like greetings, thanks, or small talk), respond naturally **without using the context**.
        3. If the context does not contain relevant information for the user's query, say "I don't know the answer based on the available context."

        Context: {context}
        Question: {question}

        Your response:
        """


    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=vectorstore.as_retriever(), 
        chain_type_kwargs={"prompt": rag_prompt},
    )

    response = qa_chain.invoke(text)
    return response['result']

if __name__ == "__main__":
    print(get_response("Give me placement stattistics of 2018"))