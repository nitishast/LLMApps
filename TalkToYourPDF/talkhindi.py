import os
import config
import streamlit as st
import fasttext as ft

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4


api_key = config.GOOGLE_API_KEY
chat_model = config.CHAT_MODEL
chunk_size = config.CHUNK_SIZE
chunk_overlap = config.CHUNK_OVERLAP
embeddings_model = config.EMBEDDINGS_MODEL

# Load fasttext model
embedding_model_path = '/Users/nitastha/Desktop/NitishFiles/Projects/wiki.hi/wiki.hi.bin'
embeddings_model = ft.load_model(embedding_model_path)


def get_text_from_txt(uploaded_files):
    """
    Extract text from uploaded TXT files.
    
    :param uploaded_files: List of uploaded TXT file objects
    :return: Concatenated text from all TXT files
    """
    text = ""
    for uploaded_file in uploaded_files:
        for line in uploaded_file.readlines():
            text += line.decode('utf-8')
    return text

def preprocess_text(text):
    """
    Preprocess the extracted text by splitting it into lines and creating document objects.
    
    :param text: The input text from TXT file extraction
    :return: List of LangChain Document objects
    """
    lines = text.splitlines()
    documents = []
    current_page = None
    line_number = 0

    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if line.startswith("--- Page"):  # Detect page headers
            try:
                current_page = int(line.split()[-2])  # Get the page number (penultimate element)
            except ValueError:
                continue  # Skip if the page number is invalid (just in case)
            line_number = 0  # Reset line number for a new page
        elif line:  # Non-empty line
            documents.append(
                Document(
                    page_content=line,
                    metadata={"page": current_page, "line_number": line_number}
                )
            )
            line_number += 1
    return documents

def get_text_chunks(documents, chunk_size=512, chunk_overlap=70):
    """
    Split documents into chunks.
    
    :param documents: List of LangChain Document objects
    :param chunk_size: Size of each text chunk
    :param chunk_overlap: Overlap between chunks
    :return: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


def get_vector_store(documents):
    """
    Creates and returns a Qdrant Vector Store.

    :param documents: List of Document objects
    :return: Qdrant Vector Store object
    """
    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(size=embeddings_model.get_dimension(), distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embeddings=embeddings_model,
    )
    
    vector_store.add_documents(documents=documents)
    return vector_store

def get_conversation_chain():
    prompt_template = """
    <s>[INST] आप एक विश्वसनीय और सटीक सहायक हैं। आपको केवल और केवल नीचे दिए गए संदर्भ के आधार पर प्रश्न का उत्तर देना है। 

            निर्देश:
            - केवल दिए गए संदर्भ से जानकारी का उपयोग करें
            - यदि संदर्भ में उत्तर नहीं मिलता है, तो स्पष्ट रूप से कहें कि "दिए गए संदर्भ में इस प्रश्न का उत्तर नहीं मिलता"
            - अपने ज्ञान या अतिरिक्त जानकारी को शामिल न करें
            - उत्तर संक्षिप्त, स्पष्ट और सीधा होना चाहिए
            - हिंदी भाषा में ही उत्तर दें

    संदर्भ: \n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=chat_model, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, vector_store):
    """Handles user input, retrieves relevant documents, and generates an answer."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Adjust 'k' as needed

    docs = retriever.get_relevant_documents(user_question)
    
    chain = get_conversation_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.markdown(
        f"""
        <div style='background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-top: 10px;'>
            <h4 style='color: #333; margin-bottom: 10px;'>📝 Question:</h4>
            <p style='font-weight: bold; color: #555; margin-bottom: 15px;'>{user_question}</p>
            <h4 style='color: #333; margin-bottom: 10px;'>💡 Answer:</h4>
            <p style='color: #444;'>{response['output_text']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def runhindi():
    st.header("Chat with Multiple TXT files using Qdrant, Langchain, and Gemini Pro")
    
    user_question = st.text_input("Please write your question?")
    st.text("Here are some sample questions:")
    st.text("What is the document about?")
    # st.text("What are the names of the artists?")

    with st.sidebar:
        st.title("Menu:")
        txt_docs = st.file_uploader("Upload your TXT files and click on submit.", accept_multiple_files=True)
        if st.button("Submit & Process", type="primary"):
            if txt_docs:
                with st.spinner("Processing..."):
                    # Extract text from TXT files
                    raw_text = get_text_from_txt(txt_docs)
                    st.text("Text extracted from TXT files.")

                    # Preprocess text
                    documents = preprocess_text(raw_text)
                    st.text("Text preprocessed into documents.")
                    
                    # Create text chunks
                    text_chunks = get_text_chunks(documents)
                    st.text("Text chunks created.")

                    # Create vector store
                    vector_store = get_vector_store(text_chunks)
                    st.text("Vector store created with Qdrant.")

                    st.session_state.vector_store = vector_store
                st.success("Processing completed successfully!")
            else:
                st.error("Please upload TXT files before processing.")
    
    if user_question and "vector_store" in st.session_state:
       user_input(user_question, st.session_state.vector_store)

