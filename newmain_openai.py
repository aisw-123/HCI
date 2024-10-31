from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

def load_api_key():
    load_dotenv()
    api_key = st.secrets["OPENAI_API_KEY"]
    return api_key

# Optimized function to extract text page-by-page from a PDF
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        yield page.extract_text()

# Function to summarize text from the PDF in batches
def summarize_pdf(text_generator, api_key):
    llm = OpenAI(api_key=api_key)
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)  # Increase chunk size
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Summarize in chunks
    summaries = []
    for text in text_generator:
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        summary = chain.run(docs)
        summaries.append(summary)
    
    return "\n".join(summaries)

# Create embeddings with optimized chunk sizes
def create_knowledge_base(text_generator, api_key):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,  # Increase chunk size to reduce the number of chunks
        chunk_overlap=100,
        length_function=len
    )

    chunks = []
    for text in text_generator:
        chunks.extend(text_splitter.split_text(text))

    embeddings = OpenAIEmbeddings(api_key=api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

def query_llm(knowledge_base, user_question, api_key):
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI(api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
    
    return response

def set_chat_styles():
    st.markdown(
        """
        <style>
        .user-bubble {
            background-color: #DCF8C6;
            border-radius: 10px;
            padding: 10px;
            margin: 5px;
            max-width: 60%;
            float: right;
            clear: both;
            color: black;
        }
        .ai-bubble {
            background-color: #F1F0F0;
            border-radius: 10px;
            padding: 10px;
            margin: 5px;
            max-width: 60%;
            float: left;
            clear: both;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    api_key = load_api_key()
    st.header("Ask About Your PDF 🤷‍♀️💬")
    set_chat_styles()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_summary' not in st.session_state:
        st.session_state.pdf_summary = None
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ''

    pdf = st.file_uploader("Upload your PDF File and Ask Questions", type="pdf")

    if pdf is not None and st.session_state.pdf_summary is None:
        text_generator = extract_text_from_pdf(pdf)
        with st.spinner('Extracting and summarizing PDF...'):
            summary = summarize_pdf(text_generator, api_key)
            st.session_state.pdf_summary = summary
            st.success('Summary generated successfully!')

        st.session_state.knowledge_base = create_knowledge_base(extract_text_from_pdf(pdf), api_key)

    if st.session_state.pdf_summary:
        st.subheader("PDF Summary:")
        st.write(st.session_state.pdf_summary)

    if st.session_state.pdf_summary:
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.markdown(f"<div class='user-bubble'><strong>You:</strong><br>{chat['question']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='ai-bubble'><strong>AI:</strong><br>{chat['response']}</div>", unsafe_allow_html=True)

        user_question = st.text_input("Please ask a question about your PDF here:", value=st.session_state.user_question, key="user_question_input")

        if st.button("Submit"):
            if user_question:
                with st.spinner('Processing your question...'):
                    response = query_llm(st.session_state.knowledge_base, user_question, api_key)

                    st.session_state.chat_history.append({
                        "question": user_question,
                        "response": response
                    })

                    st.markdown(f"<div class='user-bubble'><strong>You:</strong><br>{user_question}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='ai-bubble'><strong>AI:</strong><br>{response}</div>", unsafe_allow_html=True)

                st.session_state.user_question = ''

if __name__ == '__main__':
    main()
