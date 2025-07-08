import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.document_loaders.csv_loader import CSVLoader


# =========================
# File Handling Functions
# =========================
def get_files_text(files):
    text = ''
    for file in files:
        ext = os.path.splitext(file.name)[1].lower()
        if ext == '.pdf':
            text += get_pdf_text(file)
        elif ext == '.docx':
            text += get_docx_text(file)
        elif ext == '.csv':
            text += get_csv_text(file)
    return text


def get_pdf_text(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def get_docx_text(file):
    doc = docx.Document(file)
    return ' '.join([para.text for para in doc.paragraphs])


def get_csv_text(file):
    loader = CSVLoader(file)
    data = loader.load()
    return ' '.join([d.page_content for d in data])


# =========================
# LangChain Setup
# =========================
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain


# =========================
# Chat Handler
# =========================
def handle_userinput(user_question, openai_api_key):
    try:
        if not openai_api_key or openai_api_key.strip() == "":
            st.error("❌ OpenAI API key is missing.")
            return

        with get_openai_callback() as cb:
            response = st.session_state.conversation({'question': user_question})

        st.session_state.chat_history = response['chat_history']

        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

    except Exception:
        st.error("❌ An error occurred while processing your request. Please check your API key or try again.")


# =========================
# Main App
# =========================
def main():
    st.set_page_config(page_title='DocumentGPT by Sami')
    st.header('📄 Abdul Sami DocumentGPT')
    st.markdown("""
Upload a document (PDF, DOCX, or CSV), enter your OpenAI API key, and start chatting with your document!

> ℹ️ **How it works**  
> - Your uploaded documents and API key are never saved.  
> - All data is stored in memory and cleared on browser refresh.  
> - Works best with text-based PDFs (not scanned images).  
> 
> 🔐 [Get your OpenAI API key here](https://platform.openai.com/account/api-keys)
""")

    # Init session state
    for key in ['conversation', 'chat_history', 'processComplete']:
        if key not in st.session_state:
            st.session_state[key] = None

    # Sidebar input
    with st.sidebar:
        openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
        upload_files = st.file_uploader("📎 Upload your documents", type=['pdf', 'docx', 'csv'], accept_multiple_files=True)
        process = st.button("🚀 Process Documents")

    # Document processing
    if process:
        if not openai_api_key or openai_api_key.strip() == "":
            st.error("❌ Please enter your OpenAI API key.")
            st.stop()
        if not upload_files:
            st.error("❌ Please upload at least one document.")
            st.stop()

        file_text = get_files_text(upload_files)
        if not file_text.strip():
            st.error("⚠️ No readable text found in the uploaded documents.")
            st.stop()

        st.success("✅ Document loaded")

        text_chunks = get_text_chunks(file_text)
        st.success("✅ Text split into chunks")

        vectorstore = get_vectorstore(text_chunks)
        st.success("✅ Vector store created (in memory)")

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True
        st.session_state.chat_history = []

    # Chat interface
    if st.session_state.processComplete:
        user_question = st.chat_input("💬 Ask your question about the document")
        if user_question:
            handle_userinput(user_question, openai_api_key)


if __name__ == '__main__':
    main()
