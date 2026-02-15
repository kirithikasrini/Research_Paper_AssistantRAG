# app.py

import streamlit as st
import tempfile
from rag_pipeline import create_vectorstore, create_qa_chain


st.title("📚 Research Paper Assistant (RAG)")

uploaded_file = st.file_uploader("Upload Research Paper", type="pdf")

if uploaded_file:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF Uploaded Successfully!")

    # Create vectorstore
    with st.spinner("Processing PDF..."):
        vectorstore = create_vectorstore(pdf_path)

    # Create QA chain
    with st.spinner("Loading AI Model (this might take a moment)..."):
        qa_chain = create_qa_chain(vectorstore)

    question = st.text_input("Ask a question about the paper")

    if question:
        answer = qa_chain.run(question)
        st.write("### Answer:")
        st.write(answer)
