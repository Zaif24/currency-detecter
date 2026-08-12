"""
Ingestion pipeline for the RAG component.

Embedding model : sentence-transformers/all-MiniLM-L6-v2 (primary)
  - 384-dim embeddings, ~80MB, runs on CPU with no API cost or key required.
  - Chosen over a hosted embedding API (e.g. OpenAI/OpenRouter embeddings)
    because it keeps ingestion free, fast, and reproducible for grading
    without requiring an extra API key just to build the index.
  - Requires downloading model weights from huggingface.co on first run
    (works fine on Streamlit Cloud, which has outbound internet access).

Fallback embedding : TF-IDF (scikit-learn), used automatically if the
  sentence-transformer model cannot be downloaded (e.g. offline/restricted
  network environments such as this development sandbox). This keeps the
  pipeline runnable everywhere; embedding_backend.py records which mode
  produced the index so the README/report can note it.

Vector store : FAISS (IndexFlatIP, cosine similarity via L2-normalized vectors)
  - Chosen over Chroma for this project because it is a single dependency,
    has zero external server/process requirement, and is trivial to persist
    as a flat file for Streamlit Cloud's ephemeral filesystem.

Run:
    python -m rag.ingest
Produces:
    vectorstore/index.faiss
    vectorstore/metadata.json
    vectorstore/embedder.json   (records which embedding backend was used)
"""
import os
import json
import glob

import numpy as np
import faiss

from rag.chunking import chunk_document
from rag.embedding_backend import get_embedder

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")
STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
META_PATH = os.path.join(STORE_DIR, "metadata.json")
EMBEDDER_INFO_PATH = os.path.join(STORE_DIR, "embedder.json")


def load_documents():
    docs = []
    for path in glob.glob(os.path.join(KB_DIR, "**", "*.md"), recursive=True):
        country = os.path.basename(os.path.dirname(path))
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = f"{country}/{os.path.basename(path)}"
        docs.append((text, source, country))
    return docs


def build_index():
    os.makedirs(STORE_DIR, exist_ok=True)
    embedder = get_embedder()
    print(f"Using embedding backend: {embedder.name}")

    docs = load_documents()
    print(f"Found {len(docs)} source documents in knowledge base.")

    all_chunks = []
    for text, source, country in docs:
        chunks = chunk_document(text, source)
        for c in chunks:
            all_chunks.append({"text": c.text, "source": c.source, "country": country})

    print(f"Produced {len(all_chunks)} chunks after chunking.")

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.fit_encode(texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)  # normalize for cosine similarity via inner product

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    embedder.save(STORE_DIR)
    with open(EMBEDDER_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump({"backend": embedder.name}, f, indent=2)

    print(f"Saved FAISS index -> {INDEX_PATH}")
    print(f"Saved metadata    -> {META_PATH}")
    print(f"Embedding backend used: {embedder.name}")


if __name__ == "__main__":
    build_index()
