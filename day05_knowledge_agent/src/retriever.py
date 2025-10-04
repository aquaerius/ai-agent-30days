# src/retriever.py
"""
Simple local TF-IDF retriever with chunking and pickle persistence.
- Supports .txt and .pdf from a data directory.
- Chunks text into ~800-character windows with 200-character overlap.
- Builds a TF-IDF matrix for fast top-k retrieval.
"""

from __future__ import annotations
import os, re, pickle
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pypdf import PdfReader

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

@dataclass
class Chunk:
    doc_path: str
    chunk_id: int
    text: str

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)

def _normalize(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _chunk(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = _normalize(text)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start <= 0:
            break
    return chunks

def load_corpus(data_dir: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name.lower())[1]
            if ext not in [".txt", ".pdf"]:
                continue
            try:
                text = _read_txt(path) if ext == ".txt" else _read_pdf(path)
                if not text.strip():
                    continue
                for i, c in enumerate(_chunk(text)):
                    chunks.append(Chunk(doc_path=os.path.relpath(path, data_dir), chunk_id=i, text=c))
            except Exception:
                # Skip unreadable files gracefully
                continue
    return chunks

@dataclass
class Index:
    vectorizer: TfidfVectorizer
    matrix: np.ndarray
    chunks: List[Chunk]
    data_dir: str

def build_index(data_dir: str) -> Index:
    chunks = load_corpus(data_dir)
    texts = [c.text for c in chunks]
    if not texts:
        raise RuntimeError(f"No usable .txt or .pdf files found in {data_dir}")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_df=0.9,
        min_df=1,
        ngram_range=(1,2)
    )
    matrix = vectorizer.fit_transform(texts)
    return Index(vectorizer=vectorizer, matrix=matrix, chunks=chunks, data_dir=data_dir)

def save_index(index: Index, path: str):
    with open(path, "wb") as f:
        pickle.dump(index, f)

def load_index(path: str) -> Index:
    with open(path, "rb") as f:
        return pickle.load(f)

def retrieve(index: Index, query: str, k: int = 4) -> List[Tuple[Chunk, float]]:
    qv = index.vectorizer.transform([query])
    scores = (index.matrix @ qv.T).toarray().ravel()
    top = np.argsort(scores)[::-1][:k]
    return [(index.chunks[i], float(scores[i])) for i in top if scores[i] > 0]
