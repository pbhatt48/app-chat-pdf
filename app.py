import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import  OpenAIEmbeddings
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os


def init():
    load_dotenv()
    if os.getenv('OPENAI_API_KEY') is None or os.getenv('HUGGINGFACEHUB_API_TOKEN') is None:
        print("OPEN API OR HUGGING FACE API KEYS NOT SET")
    else:
        print("KEYS ARE SET!")

def main():
    init()
    st.header("Chat with your PDF documents ")
    st.sidebar.title("LLM ChatApp using LangChain")
    st.sidebar.markdown('''
    This is an LLM powerded chatbot build using 
    -[Streamlit] (https://streamlit.io/)''')


    ##upload pdf file
    pdf = st.file_uploader('Upload your PDF file', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        #st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        #st.write(chunks[1])
        store_name = pdf.name[:-4]
        st.write(store_name)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", 'rb') as f:
                vector_store = pickle.load(f)
            st.write("Embeddings loaded from the disk!")
        else:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            st.write('Embeddings Created!')

        query = st.text_input("Ask Question for your PDF file")
        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)



if __name__ == "__main__":
    main()


