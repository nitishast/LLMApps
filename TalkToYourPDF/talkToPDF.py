import os
import config
import google.generativeai as genai
import streamlit as st

from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

api_key = config.GOOGLE_API_KEY
chat_model = config.CHAT_MODEL
chunk_size = config.CHUNK_SIZE
chunk_overlap = config.CHUNK_OVERLAP
embeddings_model = config.EMBEDDINGS_MODEL

genai.configure(api_key=api_key)

def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunk_from_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_info(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context, just say " Answer is not in the provided context", don't provide wrong answers and additional commentary.
    Context : \n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=chat_model, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    st.text(docs)
    chain = get_conversation_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    # Create a more structured and visually appealing response display
    st.markdown("""
    <div style='background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-top: 10px;'>
        <h4 style='color: #333; margin-bottom: 10px;'>📝 Question:</h4>
        <p style='font-weight: bold; color: #555; margin-bottom: 15px;'>""" + user_question + """</p>
        <h4 style='color: #333; margin-bottom: 10px;'>💡 Answer:</h4>
        <p style='color: #444;'>""" + response["output_text"] + """</p>
    </div>
    """, unsafe_allow_html=True)

def run():
    st.header("Chat with Multiple PDF using FAISS, Vector Store, Langchain, and Gemini Pro")

    user_question = st.text_input("Please write your question?")

    # st.text("Here are some sample questions:")
    st.text("What is the document about?")
    # st.text("What are the names of the artists?")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on submit.", accept_multiple_files=True)
        if st.button("Submit & Process", type="primary"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    st.text("Text extracted from PDFs.")

                    # Create text chunks
                    text_chunks = get_chunk_from_text(raw_text)
                    st.text("Text chunks created.")

                    # Process vector information
                    get_vector_info(text_chunks)
                    st.text("Vector information processed.")

                st.success("Processing completed successfully!")
            else:
                st.error("Please upload PDF files before processing.")

