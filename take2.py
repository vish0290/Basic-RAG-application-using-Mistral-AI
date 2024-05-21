from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MISTRAL"]

def rag_test(path,user_data):
    loader = TextLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    model = ChatMistralAI(mistral_api_key=api_key)
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context: <context> {context} </context> Question: {input}""")
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": f"{user_data}"})
    return response["answer"]