# tests/test_memory.py
import os
import tempfile
from src.memory import MemoryStore

def test_add_and_recall_recent():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "mem.db")
        m = MemoryStore(db)
        m.add_interaction("hi", "hello")
        m.add_interaction("how are you?", "great")
        recent = m.recall_recent(3)
        assert len(recent) >= 2
        assert recent[-1].role == "agent"
        m.close()

def test_keyword_search_like():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "mem.db")
        m = MemoryStore(db)
        m.add_interaction("I love Python lists", "Cool!")
        m.add_interaction("Dictionaries are maps", "Indeed")
        hits = m.recall_keywords("lists", limit=5)
        assert any("lists" in t.text.lower() for t in hits)
        m.close()
