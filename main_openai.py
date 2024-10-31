from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Function to load environment variables and return OpenAI API key
def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    print("Loaded API Key:", api_key)  # Debugging line
    return api_key

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to summarize the extracted text from the PDF
def summarize_pdf(text, api_key):
    llm = OpenAI(openai_api_key=api_key)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    
    # Create Document objects
    docs = [Document(page_content=t) for t in texts]
    
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    
    return summary

# Function to create embeddings and return a knowledge base from the extracted PDF chunks
def create_knowledge_base(text, api_key):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

# Function to handle LLM call and generate a response for the given question
def query_llm(knowledge_base, user_question, api_key):
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI(openai_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
    
    return response

# Custom CSS for chat bubbles
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

# Main function for handling the app workflow
def main():
    api_key = "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"
    st.header("Ask About Your PDF 🤷‍♀️💬")

    # Set chat bubble styles
    set_chat_styles()

    # Initialize session state for chat history, summary, and user question
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_summary' not in st.session_state:
        st.session_state.pdf_summary = None
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ''  # Initialize the text input state

    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF File and Ask Questions", type="pdf")

    if pdf is not None and st.session_state.pdf_summary is None:
        # Extract text from the PDF
        with st.spinner('Extracting text from PDF...'):
            text = extract_text_from_pdf(pdf)
            st.success('Text extracted successfully!')

        # Generate a detailed summary of the PDF
        with st.spinner('Summarizing the document...'):
            summary = summarize_pdf(text, api_key)
            st.session_state.pdf_summary = summary
            st.success('Summary generated successfully!')

        # Create knowledge base from the extracted text (after summarization)
        st.session_state.knowledge_base = create_knowledge_base(text, api_key)

    # Keep the summary displayed at the top
    if st.session_state.pdf_summary:
        st.subheader("PDF Summary:")
        st.write(st.session_state.pdf_summary)

    if st.session_state.pdf_summary:
        # Display chat history with bubbles
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.markdown(f"<div class='user-bubble'><strong>You:</strong><br>{chat['question']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='ai-bubble'><strong>AI:</strong><br>{chat['response']}</div>", unsafe_allow_html=True)

        # Get user input and handle LLM query
        user_question = st.text_input("Please ask a question about your PDF here:", value=st.session_state.user_question, key="user_question_input")

        if st.button("Submit"):
            if user_question:  # Only process if there's a question
                with st.spinner('Processing your question...'):
                    response = query_llm(st.session_state.knowledge_base, user_question, api_key)

                    # Add the current question and response to the chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "response": response
                    })

                    # Display the latest chat bubble
                    st.markdown(f"<div class='user-bubble'><strong>You:</strong><br>{user_question}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='ai-bubble'><strong>AI:</strong><br>{response}</div>", unsafe_allow_html=True)

                # Clear the input field after submission
                st.session_state.user_question = ''  # Reset the session state for user input

if __name__ == '__main__':
    main()
