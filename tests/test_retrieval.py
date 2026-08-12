"""
Basic tests. Run with: PYTHONPATH=. pytest tests/ -v
Requires the vector store to already be built: python -m rag.ingest
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.chunking import chunk_document
from rag.retriever import retrieve


def test_chunking_produces_chunks():
    text = "# Title\n\nParagraph one.\n\nParagraph two."
    chunks = chunk_document(text, source="test.md")
    assert len(chunks) >= 1
    assert all(c.source == "test.md" for c in chunks)


def test_retrieve_returns_results():
    results = retrieve("security features of banknotes", k=3)
    assert len(results) > 0
    assert all("source" in r and "score" in r for r in results)


def test_retrieve_country_filter():
    results = retrieve("currency history", k=3, country_filter="japan")
    assert all(r["country"] == "japan" for r in results)


def test_retrieve_relevant_top_result():
    results = retrieve("Thai baht float 1997 Asian financial crisis", k=1)
    assert results[0]["country"] == "thailand"
