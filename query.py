import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from utils import load_pickle
from config import (
    INDEX_FILE, META_FILE, VECTORS_FILE,
    HF_MODEL, TOP_K,
    GENERATION_BACKEND, OPENAI_API_KEY, OPENAI_MODEL
)

# Load vectors + metadata
index = faiss.read_index(INDEX_FILE)
meta = load_pickle(META_FILE)
vectors = np.load(VECTORS_FILE)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------- LOCAL GENERATION --------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)

def local_generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def openai_generate(prompt):
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer based on the context provided."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# -------- RETRIEVAL --------
def embed_query(q):
    return embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)

def retrieve(q):
    qv = embed_query(q)
    D, I = index.search(qv, TOP_K)
    return I[0]

def get_context(idxs):
    """Retrieve context from pre-stored chunks in metadata."""
    ctx = []
    for i in idxs:
        ctx.append(meta[i]["chunk_text"])
    return "\n\n".join(ctx)

def answer(q):
    retrieved = retrieve(q)
    ctx = get_context(retrieved)

    prompt = (
        f"Context:\n{ctx}\n\n"
        f"Question: {q}\nAnswer:"
    )

    if GENERATION_BACKEND == "openai":
        return openai_generate(prompt)
    else:
        return local_generate(prompt)

# -------- CLI --------
if __name__ == "__main__":
    while True:
        q = input("\nQuery: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n=== ANSWER ===\n")
        print(answer(q))
