import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import  OpenAIEmbeddings
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from htmlTemplates import css, bot_template, user_template
from langchain.chat_models import ChatOpenAI



def init():
    load_dotenv()
    if os.getenv('OPENAI_API_KEY') is None or os.getenv('HUGGINGFACEHUB_API_TOKEN') is None:
        print("OPEN API OR HUGGING FACE API KEYS NOT SET")
    else:
        print("KEYS ARE SET!")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200,
    #                                       length_function=len)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750,
                                                   chunk_overlap=200,
                                                   )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    ## for hugging face embeddings
    #embeddings = HuggingFaceEmbeddings(model_name=os.getenv('EMBEDDING_MODEL_NAME'))

    ## For OpenAI embeddings:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):

    ##For hugging face model
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={'temperature':0.5, 'max_length':512})
    ## for OpenAI
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            #st.write(message)
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            #st.write(message)
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    init()
    st.set_page_config("Chat with multiple PDFs", page_icon=":books")
    st.header("Chat with multiple PDFs :books: ")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat = None

    user_question = st.text_input("Ask a question from your documents")
    if user_question:
        handle_user_input(user_question)


    with st.sidebar:
        st.subheader("Documents")
        st.title("LLM Chatapp using LangChain")
        pdf_docs = st.file_uploader("Upload your pdfs here and Click on Process", accept_multiple_files=True)
        #st.button("Process")


        if st.button("Process"):
            with st.spinner("Processsing...."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore=get_vector_store(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Done!")





if __name__ =="__main__":
    main()