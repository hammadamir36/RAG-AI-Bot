"""
Flask Web Application for RAG System
Allows users to upload documents and query them
"""
import os
import json
import shutil
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai

from utils import (
    load_document, chunk_text, save_pickle, load_pickle,
    list_docs
)
from config import (
    INDEX_DIR, INDEX_FILE, META_FILE, VECTORS_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP, ST_MODEL_NAME,
    HF_MODEL, TOP_K, GENERATION_BACKEND, OPENAI_API_KEY, OPENAI_MODEL
)

# Flask app setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'data'

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'txt', 'md', 'csv', 'json', 'docx', 'xlsx', 'html', 'doc', 'ppt', 'pptx'
}

# Global state
index = None
meta = None
vectors = None
embedder = None
model = None
tokenizer = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_rag_index():
    """Load existing RAG index if it exists."""
    global index, meta, vectors, embedder
    
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE) and os.path.exists(VECTORS_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            meta = load_pickle(META_FILE)
            vectors = np.load(VECTORS_FILE)
            embedder = SentenceTransformer(ST_MODEL_NAME)
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    return False

def initialize_models():
    """Initialize embedding and generation models."""
    global embedder, model, tokenizer
    
    if embedder is None:
        embedder = SentenceTransformer(ST_MODEL_NAME)
    
    if GENERATION_BACKEND == "local" and model is None:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)

def embed_query(q):
    """Embed a query."""
    if embedder is None:
        initialize_models()
    return embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)

def retrieve(q):
    """Retrieve top-k relevant chunks."""
    if index is None:
        return []
    
    qv = embed_query(q)
    D, I = index.search(qv, TOP_K)
    return I[0]

def get_context(idxs):
    """Get context from retrieved chunks."""
    if meta is None:
        return ""
    
    ctx = []
    for i in idxs:
        if i < len(meta):
            ctx.append(meta[i]["chunk_text"])
    return "\n\n".join(ctx)

def local_generate(prompt):
    """Generate answer using local model."""
    if model is None or tokenizer is None:
        initialize_models()
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output = model.generate(**inputs, max_new_tokens=120)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating response: {str(e)}"

def openai_generate(prompt):
    """Generate answer using OpenAI."""
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured"
    
    try:
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
    except Exception as e:
        return f"Error with OpenAI: {str(e)}"

def answer_query(q):
    """Answer a query using RAG."""
    if index is None:
        return "No documents indexed yet. Please upload documents first."
    
    try:
        retrieved = retrieve(q)
        ctx = get_context(retrieved)
        
        if not ctx:
            return "No relevant context found in documents."
        
        prompt = (
            f"Context:\n{ctx}\n\n"
            f"Question: {q}\nAnswer:"
        )
        
        if GENERATION_BACKEND == "openai":
            return openai_generate(prompt)
        else:
            return local_generate(prompt)
    except Exception as e:
        return f"Error processing query: {str(e)}"

def build_index():
    """Build FAISS index from documents in data folder."""
    global index, meta, vectors
    
    try:
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # Load documents
        docs = list_docs("data")
        if not docs:
            return False, "No documents found in data folder."
        
        # Chunk documents
        chunks = []
        metadata = []
        
        for path in docs:
            text = load_document(path)
            if not text.strip():
                continue
            
            doc_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, ch in enumerate(doc_chunks):
                chunks.append(ch)
                metadata.append({
                    "source": path,
                    "chunk_id": i,
                    "chunk_text": ch
                })
        
        if not chunks:
            return False, "No content extracted from documents."
        
        # Embed chunks
        initialize_models()
        vectors_array = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        
        # Build FAISS index
        index = faiss.IndexFlatIP(vectors_array.shape[1])
        index.add(vectors_array)
        
        # Save index
        faiss.write_index(index, INDEX_FILE)
        np.save(VECTORS_FILE, vectors_array)
        save_pickle(metadata, META_FILE)
        
        meta = metadata
        vectors = vectors_array
        
        return True, f"Index built successfully with {len(chunks)} chunks from {len(docs)} documents."
    
    except Exception as e:
        return False, f"Error building index: {str(e)}"

@app.route('/')
def index_route():
    """Home page."""
    is_indexed = index is not None
    return render_template('index.html', is_indexed=is_indexed)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload."""
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'message': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        uploaded_count = 0
        errors = []
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_count += 1
            else:
                errors.append(f"File '{file.filename}' not allowed or invalid")
        
        return jsonify({
            'success': True,
            'message': f'Uploaded {uploaded_count} file(s)',
            'errors': errors
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/build-index', methods=['POST'])
def build_index_route():
    """Build FAISS index from uploaded documents."""
    success, message = build_index()
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/query', methods=['POST'])
def query_route():
    """Process a query."""
    try:
        data = request.json
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({'success': False, 'message': 'Query cannot be empty'}), 400
        
        if index is None:
            return jsonify({'success': False, 'message': 'No documents indexed yet'}), 400
        
        answer = answer_query(query_text)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'query': query_text
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get RAG system status."""
    if index is not None and meta is not None:
        num_chunks = len(meta)
        num_docs = len(set(m['source'] for m in meta))
        return jsonify({
            'indexed': True,
            'num_chunks': num_chunks,
            'num_documents': num_docs
        })
    else:
        return jsonify({
            'indexed': False,
            'num_chunks': 0,
            'num_documents': 0
        })

@app.route('/clear-index', methods=['POST'])
def clear_index_route():
    """Clear the index and data."""
    global index, meta, vectors
    
    try:
        index = None
        meta = None
        vectors = None
        
        # Remove index files
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(META_FILE):
            os.remove(META_FILE)
        if os.path.exists(VECTORS_FILE):
            os.remove(VECTORS_FILE)
        
        # Clear data folder
        if os.path.exists('data'):
            shutil.rmtree('data')
        
        return jsonify({'success': True, 'message': 'Index and data cleared'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # Load existing index if available
    load_rag_index()
    
    # Initialize models
    initialize_models()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
