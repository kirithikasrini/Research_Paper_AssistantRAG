from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate
import streamlit as st
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



@st.cache_resource
def create_vectorstore(_pdf_path, file_id):
    # Load PDF
    loader = PyPDFLoader(_pdf_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Store in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


@st.cache_resource
def create_qa_chain(_vectorstore):

    # Load free LLM
    # Load slightly larger but more capable free LLM
    model_name = "MBZUAI/LaMini-Flan-T5-248M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Create customized prompt template for better instructions
    template = """Use the following pieces of context to explain the topic or answer the question. 
    Provide a clear and detailed response. If you don't know the answer, just say that you don't know.

    Context: {context}

    Question: {question}

    Explanation:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain
