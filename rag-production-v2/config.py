"""
config.py — Chargement centralisé de la configuration depuis .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PDF_DIR       = Path(os.getenv("PDF_DIR", "./data/pdfs"))
QDRANT_URL    = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION    = os.getenv("COLLECTION_NAME", "prod_documents")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL     = os.getenv("LLM_MODEL", "mistral-nemo:12b")
EMBED_DIM     = int(os.getenv("EMBED_DIM", "768"))
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K         = int(os.getenv("TOP_K", "15"))
