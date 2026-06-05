"""Configuration for the RAG assistant."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Current Models
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "llama3"

# Retrieval Settings
TOP_K = 4
FETCH_K = 10

# Chunking Settings
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300