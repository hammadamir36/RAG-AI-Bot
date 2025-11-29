# ingest.py
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import list_docs, load_document, chunk_text, save_pickle
from config import (
    INDEX_DIR, INDEX_FILE, META_FILE, VECTORS_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP, ST_MODEL_NAME
)

os.makedirs(INDEX_DIR, exist_ok=True)

def main():
    print("[1] Loading documents...")
    docs = list_docs("data")
    if not docs:
        print("⚠️ No documents found inside /data folder.")
        return

    print("[2] Chunking...")
    chunks = []
    metadata = []

    for path in docs:
        text = load_document(path)
        doc_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, ch in enumerate(doc_chunks):
            chunks.append(ch)
            metadata.append({"source": path, "chunk_id": i, "chunk_text": ch})

    print(f"Total chunks: {len(chunks)}")

    print("[3] Embedding using SentenceTransformer...")
    model = SentenceTransformer(ST_MODEL_NAME)
    vectors = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    print("[4] Building FAISS index...")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    print("[5] Saving index...")
    faiss.write_index(index, INDEX_FILE)
    np.save(VECTORS_FILE, vectors)
    save_pickle(metadata, META_FILE)

    print("✅ Done! RAG index ready.")

if __name__ == "__main__":
    main()
