# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

TOP_K = int(os.getenv("TOP_K", 5))

GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "local")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

INDEX_DIR = "index"
INDEX_FILE = f"{INDEX_DIR}/index.faiss"
META_FILE = f"{INDEX_DIR}/meta.pkl"
VECTORS_FILE = f"{INDEX_DIR}/vectors.npy"
