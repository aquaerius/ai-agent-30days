# tests/test_retriever.py
import os, tempfile, shutil
from src.retriever import build_index, retrieve

def test_build_and_search_txt(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("Python lists are ordered collections. They support indexing and slicing.")
    (data / "b.txt").write_text("Dictionaries map keys to values. Keys should be immutable.")

    idx = build_index(str(data))
    results = retrieve(idx, "What are Python lists?", k=3)
    assert len(results) >= 1
    top_chunk, score = results[0]
    assert "lists" in top_chunk.text.lower()
    assert score > 0
