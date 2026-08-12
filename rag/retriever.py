"""
Retriever used by the Research/RAG Agent.
Loads the FAISS index + metadata built by rag/ingest.py and exposes a
simple `retrieve(query, k)` API returning ranked chunks with similarity scores.
Automatically uses whichever embedding backend (sentence-transformers or
TF-IDF fallback) was used to build the index, recorded in embedder.json.
"""
import os
import json
import functools

import numpy as np
import faiss

from rag.embedding_backend import load_embedder_for_query

STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
META_PATH = os.path.join(STORE_DIR, "metadata.json")
EMBEDDER_INFO_PATH = os.path.join(STORE_DIR, "embedder.json")


@functools.lru_cache(maxsize=1)
def _get_embedder():
    with open(EMBEDDER_INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    return load_embedder_for_query(STORE_DIR, info["backend"])


@functools.lru_cache(maxsize=1)
def _get_index_and_meta():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "Vector index not found. Run `python -m rag.ingest` first."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def retrieve(query: str, k: int = 3, country_filter: str = None):
    """
    Returns a list of dicts: {text, source, country, score}
    ranked by cosine similarity (highest first).
    """
    embedder = _get_embedder()
    index, meta = _get_index_and_meta()

    q_emb = np.asarray(embedder.encode([query]), dtype=np.float32)
    faiss.normalize_L2(q_emb)

    fetch_k = k * 4 if country_filter else k
    scores, ids = index.search(q_emb, min(fetch_k, len(meta)))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = meta[idx]
        if country_filter and chunk["country"] != country_filter:
            continue
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "country": chunk["country"],
            "score": float(score),
        })
        if len(results) >= k:
            break

    return results
