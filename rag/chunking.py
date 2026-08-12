"""
Chunking strategy for the knowledge base.

Strategy: paragraph-aware fixed-size chunking.
- We split each markdown document into paragraphs (blank-line separated).
- Paragraphs are then packed into chunks of up to CHUNK_SIZE characters,
  with CHUNK_OVERLAP characters carried over between consecutive chunks
  to preserve context across chunk boundaries.
- This keeps semantically related sentences together (unlike naive
  character-count slicing) while bounding chunk size for embedding quality.

For this project's short (~150-300 word) source documents, most files end up
as a single chunk, which is intentional: each document already covers one
coherent sub-topic (e.g. "LKR security features"), so further splitting would
hurt retrieval precision rather than help it.
"""
from dataclasses import dataclass
from typing import List

CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 100     # characters


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_document(text: str, source: str) -> List[Chunk]:
    paragraphs = split_into_paragraphs(text)
    chunks: List[Chunk] = []
    buffer = ""
    idx = 0

    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= CHUNK_SIZE:
            buffer = (buffer + "\n" + para).strip()
        else:
            if buffer:
                chunks.append(Chunk(text=buffer, source=source, chunk_id=idx))
                idx += 1
                # carry overlap forward
                buffer = buffer[-CHUNK_OVERLAP:] + "\n" + para
            else:
                buffer = para

    if buffer:
        chunks.append(Chunk(text=buffer, source=source, chunk_id=idx))

    return chunks
