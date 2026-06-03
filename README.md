#  RAG-Powered LLM Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over your documents using local LLMs. Built with FastAPI, LangChain, and Ollama.

## ✨ Features

- **PDF Document Processing**: Upload and automatically index PDF documents with OCR support
- **Intelligent Retrieval**: Uses MMR (Maximal Marginal Relevance) for diverse and relevant context retrieval
- **Context-Aware Responses**: Powered by Llama 3 via Ollama for accurate, document-grounded answers
- **REST API**: FastAPI-based endpoints for easy integration
- **Vector Database**: ChromaDB for efficient similarity search
- **Source Attribution**: Every answer includes source references with file names and page numbers
- **Local Embeddings**: Uses HuggingFace sentence-transformers for privacy-focused embeddings

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   PDF Docs  │ ───► │   Ingestion  │ ───► │  ChromaDB   │
└─────────────┘      │   Pipeline   │      │  (Vectors)  │
                     └──────────────┘      └─────────────┘
                                                    │
                                                    ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│    User     │ ───► │   FastAPI    │ ───► │  Retrieval  │
│   Question  │      │   Endpoints  │      │  (MMR K=8)  │
└─────────────┘      └──────────────┘      └─────────────┘
                                                    │
                                                    ▼
                     ┌──────────────┐      ┌─────────────┐
                     │   Response   │ ◄─── │  Llama 3    │
                     │  + Sources   │      │   (Ollama)  │
                     └──────────────┘      └─────────────┘
```

##  Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed with Llama 3 model
- 4GB+ RAM recommended

##  Quick Start

### 1. Install Ollama and Pull Llama 3

```bash
# Install Ollama from https://ollama.ai/
# Then pull the model
ollama pull llama3
```

### 2. Clone and Setup

```bash
git clone https://github.com/yourusername/RAG_Powered_LLM_Assistance.git
cd RAG_Powered_LLM_Assistance

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add Your Documents

```bash
# Place PDF files in the data/documents folder
mkdir -p data/documents
# Copy your PDFs here
```

### 4. Index Documents

```bash
# Run the ingestion script
python -m app.ingest
```

### 5. Start the API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation.

Or use curl:

```bash
# Upload a document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@path/to/your/document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics in the document?"}'
```

## 📁 Project Structure

```
RAG_Powered_LLM_Assistance/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── main.py            # FastAPI application
│   ├── ingest.py          # Document ingestion pipeline
│   └── rag.py             # RAG query logic
├── data/
│   └── documents/         # Your PDF documents (gitignored)
├── chroma_db/             # Vector database storage (gitignored)
├── .venv/                 # Virtual environment (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔌 API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "RAG Assistant Running"
}
```

### `POST /upload`
Upload and index a PDF document.

**Request:**
- `file`: PDF file (multipart/form-data)

**Response:**
```json
{
  "message": "PDF uploaded and indexed",
  "pages": 42,
  "chunks": 156
}
```

### `POST /ask`
Ask a question about your documents.

**Request:**
```json
{
  "question": "What is the main topic of the document?"
}
```

**Response:**
```json
{
  "question": "What is the main topic of the document?",
  "answer": "The document primarily discusses...",
  "sources": [
    {
      "file": "document.pdf",
      "page": 1,
      "content": "Relevant excerpt from the document..."
    }
  ]
}
```

## ⚙️ Configuration

Edit `app/config.py` to customize:

```python
# Directory paths
DATA_DIR = BASE_DIR / "data" / "documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Model settings (for future OpenAI integration)
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
```

## 🔧 Advanced Usage

### Running Standalone CLI

You can also run the RAG system directly from command line:

```bash
python -m app.rag
```

This starts an interactive Q&A session where you can ask questions and see detailed retrieval results.

### Customizing Retrieval

In `app/rag.py`, adjust the MMR parameters:

```python
docs = vector_store.max_marginal_relevance_search(
    query=question,
    k=8,        # Number of documents to return
    fetch_k=20  # Number of documents to fetch before MMR
)
```

### Customizing Chunking

In `app/ingest.py`, modify the text splitter:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Characters per chunk
    chunk_overlap=200   # Overlap between chunks
)
```

## 🛠️ Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern, fast web framework
- **[LangChain](https://python.langchain.com/)**: LLM application framework
- **[ChromaDB](https://www.trychroma.com/)**: Vector database
- **[Ollama](https://ollama.ai/)**: Local LLM runtime (Llama 3)
- **[HuggingFace Transformers](https://huggingface.co/)**: Embeddings (all-MiniLM-L6-v2)
- **[Unstructured](https://unstructured.io/)**: PDF parsing with OCR support

## 🎯 Use Cases

- **Document Q&A**: Query large document collections
- **Research Assistant**: Extract insights from academic papers
- **Knowledge Base**: Build a searchable company knowledge base
- **Study Aid**: Get answers from textbooks and course materials
- **Legal/Compliance**: Search through contracts and policies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- LangChain team for the excellent RAG framework
- Ollama team for making local LLMs accessible
- HuggingFace for open-source embeddings
- ChromaDB team for the vector database

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ **If you find this project helpful, please give it a star!** ⭐
