# 📚 Research Paper Assistant (RAG)

A simple **RAG (Retrieval-Augmented Generation)** application that allows you to chat with your research papers (PDFs) using free, open-source AI models.

## 🚀 Features

-   **Upload & Analyze:** Upload any PDF research paper.
-   **Smart Chunking:** Splits long documents into manageable pieces for better understanding.
-   **Vector Search:** Uses **FAISS** and **HuggingFace Embeddings** to find the most relevant sections.
-   **AI QA:** Generates answers using the free **Google Flan-T5 Small** model.
-   **Optimized Performance:** Includes caching to speed up repeated queries and spinners for better user feedback.

## 🛠️ Technology Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **Orchestration:** [LangChain](https://www.langchain.com/)
-   **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
-   **LLM Model:** [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) via HuggingFace Transformers.

## 📦 Installation

1.  **Clone the repository** (if applicable) or download the files.
2.  **Install the required Python packages**:

    ```bash
    pip install streamlit langchain langchain-community faiss-cpu transformers sentence-transformers torch
    ```

## ▶️ Usage

1.  Navigate to the project directory in your terminal:
    ```bash
    cd "ResearchAssistant"
    ```
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  The app will open in your browser.
    -   Upload a PDF using the sidebar/uploader.
    -   Wait for the "Processing PDF..." and "Loading AI Model..." steps to complete.
    -   Type your question in the text box and get an answer!

## 📂 Project Structure

-   `app.py`: The main Streamlit application file handling the user interface.
-   `rag_pipeline.py`: Contains the logic for processing the PDF, creating the vector store, and running the QA chain.

## ⚠️ Note on Model Performance

This project uses `google/flan-t5-small` which is a lightweight model designed to run on standard CPUs without a GPU. While fast and free, its answers might be brief. For more complex reasoning, consider upgrading to larger models like `flan-t5-base` or `flan-t5-large` if your hardware allows.
