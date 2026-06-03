"""Configuration for the RAG assistant."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Model and API settings (fill in or override via env vars)
OPENAI_API_KEY = None
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
