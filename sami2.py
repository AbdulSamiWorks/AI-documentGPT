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
from langchain.document_loaders.csv_loader import CSVLoader

openai_key = st.secrets['OPENAI_API_KEY']


def get_files_text(files):
    text = ''

    for file in files:
        split_tup = os.path.splitext(file.name)
        file_extension = split_tup[1]
        if file_extension == '.pdf':
            text += get_pdf_text(file)

        elif file_extension == '.docx':
            text += get_docx_text(file)

        elif file_extension == '.csv':
            text += get_csv_text(file)

    return text


def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )

    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_vectorstore(text_chunks):

    embaddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    knowladge_base = FAISS.from_texts(text_chunks, embaddings)

    return knowladge_base


def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key,
                     model='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
         memory=memory
    )

    return conversation_chain


def handle_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=(str(i)))

# file Reading in different Format


def get_pdf_text(file):
    text = ""

    pdf_reader = PdfReader(file)

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def get_docx_text(file):
    doc = docx.Document(file)
    all_text = []
    for docPara in doc.paragraphs:
        all_text += all_text.append(docPara.text)

    text = ' '.join(all_text)

    return text


def get_csv_text(file):
    loader = CSVLoader(file)
    data = loader.load()

    return data


def main():

    #load_dotenv()
    st.set_page_config(page_title='DocumentGPT by sami')
    st.header('Abdul Sami DocumentGPT')

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    if 'processComplete' not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        upload_files = st.file_uploader('Upload Your Files', type=[
                                        'pdf'], accept_multiple_files=True)
        openai_api_key = openai_key

        process = st.button('Process')

    if process:
        if not openai_api_key:
            st.info('Please Upload Your api key')
            st.stop()

        file_text = get_files_text(upload_files)
        st.write('File Loaded...')

        text_chunks = get_text_chunks(file_text)
        st.write('File Chuks Created...')

        vector_store = get_vectorstore(text_chunks)
        st.write('Vector Store Created...')

        st.session_state.conversation = get_conversation_chain(
            vector_store, openai_api_key)

        st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input('Ask Your Question About Your Document')
        if user_question:
           handle_userinput(user_question)




if __name__ == '__main__':
    main()
