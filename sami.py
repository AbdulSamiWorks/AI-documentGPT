import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer


opnai_key = st.secrets['OPENAI_API_KEY']


def main():

    st.set_page_config(page_title="Abdul Sami Document Chat model")
    st.header("DocumentGPT by Sami")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload Your Files", type=[
                                          'pdf'], accept_multiple_files=True)
        openai_key = opnai_key
        process = st.button('Process')

    if process:
        if not openai_key:
            st.info("Please Upload Your Open AI Key")
            st.stop()

        files_text = get_file_text(uploaded_files)
        st.write("File Loaded")

        text_chunks = get_text_chunks(files_text)
        st.write("File Chunks Created")

        vectorstore = get_vectorstore(text_chunks)
        st.write("Vector Store Crested")

        st.session_state.conversation = get_conservation_chain(
            vectorstore, openai_key)

        st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)


def get_file_text(uploaded_files):
    text = ''
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extention = split_tup[1]
        if file_extention == '.pdf':
            text += get_pdf_text(uploaded_file)

    return text


def get_text_chunks(text):

    text_spillter = CharacterTextSplitter(
        separator='/n',
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunk = text_spillter.split_text(text)
    return chunk


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    knowladge_base = FAISS.from_texts(text_chunks, embeddings)

    return knowladge_base


def get_conservation_chain(vectorstore, openai_key):

    llm = ChatOpenAI(openai_api_key=openai_key,
                     model='gpt-3.5-turbo', temperature=0)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conservation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conservation_chain


def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))


if __name__ == '__main__':
    main()
