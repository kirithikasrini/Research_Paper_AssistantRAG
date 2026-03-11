# 📚 Research Paper Assistant (RAG)

A powerful **RAG (Retrieval-Augmented Generation)** application that allows you to chat with your research papers (PDFs) using local, open-source AI models optimized for CPU performance.

## 🚀 Features

-   **Upload & Analyze:** Upload any PDF (Optimized for research papers).
-   **Smart Context:** Uses **FAISS** vector search with `all-MiniLM-L6-v2` embeddings for fast and accurate document retrieval.
-   **Instruction-Tuned AI:** Generates clear explanations using the **LaMini-Flan-T5** model, which is specifically trained for better instruction following.
-   **Optimized for Laptops:** Implementation includes CPU-specific optimizations and response capping to ensure reasonable speeds on standard hardware.
-   **Advanced Caching:** Uses Streamlit's `cache_resource` to remember your PDF and models, so you don't have to wait twice.

## 🛠️ Technology Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **Orchestration:** [LangChain](https://www.langchain.com/)
-   **Vector Database:** [FAISS-cpu](https://github.com/facebookresearch/faiss)
-   **Embedding Model:** `all-MiniLM-L6-v2` (Fast & Accurate)
-   **LLM Model:** [MBZUAI/LaMini-Flan-T5-248M](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M) (Superior to standard T5-Small)

## 📦 Installation

1.  **Clone the repository** (if applicable) or download the files.
2.  **Install the required Python packages**:

    ```bash
    pip install streamlit langchain langchain-community faiss-cpu transformers sentence-transformers torch
    ```

## ⚙️ Configuration & Performance Fixes

This project includes custom configurations to handle common Streamlit/Torch incompatibilities:
-   **File Watcher Disabled:** The `.streamlit/config.toml` file is set to `fileWatcherType = "none"` to prevent crashes with PyTorch's dynamic class loading.
-   **Cache Stability:** Fixed `UnhashableParamError` by using proper Streamlit caching decorators with ignored internal parameters.

## ▶️ Usage

1.  Navigate to the project directory in your terminal.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  **First Run Note:** On the very first run, the app will download approximately 1GB of AI models. This will only happen once.
4.  **Chatting:**
    -   Upload a PDF.
    -   Wait for processing to finish.
    -   Ask questions like *"Explain the methodology used here"* or *"Summarize the results."*

## 📂 Project Structure

-   `app.py`: The user interface and file management.
-   `rag_pipeline.py`: The core RAG logic (PDF Loading -> Chunking -> Vector Store -> LLM Pipeline).
-   `.streamlit/config.toml`: Performance and stability settings.

## ⚠️ Performance Tips

-   **Processing Time:** Since the AI runs on your CPU, expect 30-60 seconds for complex answers.
-   **Accuracy:** If the answer is too short, try asking more specific questions.
-   **Privacy:** All processing is done **entirely on your machine**. No data is sent to external AI servers.
