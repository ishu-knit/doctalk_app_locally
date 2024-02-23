# streamlit

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS    
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def main():
    st.title("Document Search and Question Answering")

    # File upload widget
    doc_reader = st.file_uploader("Upload a PDF file", type=["pdf"])

    if doc_reader:
        raw_text = ''
        for i, page in enumerate(PdfReader(doc_reader).pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # Splitting up the text into smaller chunks for indexing
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)



        # Initializing OpenAI Embeddings
        model = Ollama(model="zephyr",base_url = 'http://127.0.0.1:11434')
        try:
            # embeddings = OpenAIEmbeddings(api_key=api_key)
            ollama_embeddings = OllamaEmbeddings(model="zephyr",base_url = 'http://127.0.0.1:11434')

        except Exception as e:
            st.warning(f"Failed to initialize ollama_embeddings: {str(e)}")
            st.stop()

        # Building FAISS index
        try:
            vectorstore = FAISS.from_texts(texts, ollama_embeddings)
        except Exception as e:
            st.warning(f"Failed to build FAISS index: {str(e)}")
            st.stop()

        # Loading question answering chain
        try:

            chain = RetrievalQA.from_chain_type(model,retriever=vectorstore.as_retriever())

        except Exception as e:
            st.warning(f"Failed to load question answering chain: {str(e)}")
            st.stop()

        # User query input
        query = st.text_input('Type your query here... then press enter')

        if query:
            # Performing search and question answering
            try:

                result = chain.run({"query":query})
                st.write(result)
            except Exception as e:
                st.warning(f"Failed to perform search and question answering: {str(e)}")

main()


# start with "streamlit run locally_faiss.py"
