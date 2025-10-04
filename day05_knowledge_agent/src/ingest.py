# src/ingest.py
"""
Builds and saves the TF-IDF index from the ./data directory.
Usage:
  python src/ingest.py
"""
import os
from retriever import build_index, save_index

def main():
    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, "data")
    out_path = os.path.join(base, "tfidf.index.pkl")
    idx = build_index(data_dir)
    save_index(idx, out_path)
    print(f"Indexed {len(idx.chunks)} chunks from {data_dir}")
    print(f"Saved index to {out_path}")

if __name__ == "__main__":
    main()
