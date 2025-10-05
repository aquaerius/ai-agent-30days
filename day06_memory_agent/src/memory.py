# src/memory.py
"""
SQLite-backed conversation memory for an AI agent.
- Stores user/agent messages with timestamps.
- Recalls recent turns and keyword matches.
- Falls back to LIKE-based search if FTS5 isn't available.
"""

from __future__ import annotations
import os
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory.db")

@dataclass
class Turn:
    role: str
    text: str
    ts: str

class MemoryStore:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        # Base table
        c.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT CHECK(role IN ('user','agent')) NOT NULL,
            text TEXT NOT NULL,
            ts   TEXT NOT NULL
        );
        """)
        # Attempt to create FTS5 virtual table for fast search
        try:
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(text, content='turns', content_rowid='id');")
            c.execute("INSERT INTO turns_fts(rowid, text) SELECT id, text FROM turns WHERE id NOT IN (SELECT rowid FROM turns_fts);")
            self.conn.commit()
            self.has_fts = True
        except sqlite3.OperationalError:
            self.conn.rollback()
            self.has_fts = False

        self.conn.commit()

    def add_interaction(self, user_text: str, agent_text: str):
        now = datetime.utcnow().isoformat()
        c = self.conn.cursor()
        # Insert user then agent
        c.execute("INSERT INTO turns(role, text, ts) VALUES (?, ?, ?)", ("user", user_text, now))
        user_id = c.lastrowid
        c.execute("INSERT INTO turns(role, text, ts) VALUES (?, ?, ?)", ("agent", agent_text, now))
        agent_id = c.lastrowid

        if self.has_fts:
            c.execute("INSERT INTO turns_fts(rowid, text) VALUES (?, ?)", (user_id, user_text))
            c.execute("INSERT INTO turns_fts(rowid, text) VALUES (?, ?)", (agent_id, agent_text))

        self.conn.commit()

    def recall_recent(self, k: int = 6) -> List[Turn]:
        c = self.conn.cursor()
        c.execute("SELECT role, text, ts FROM turns ORDER BY id DESC LIMIT ?", (k,))
        rows = [Turn(r["role"], r["text"], r["ts"]) for r in c.fetchall()]
        rows.reverse()
        return rows

    def recall_keywords(self, query: str, limit: int = 6) -> List[Turn]:
        c = self.conn.cursor()
        if self.has_fts:
            # Simple FTS query (tokenized)
            c.execute("""
            SELECT t.role, t.text, t.ts
            FROM turns_fts f
            JOIN turns t ON t.id = f.rowid
            WHERE f.text MATCH ?
            ORDER BY t.id DESC LIMIT ?;
            """, (query, limit))
        else:
            # Fallback LIKE search
            like = f"%{query}%"
            c.execute("""
            SELECT role, text, ts
            FROM turns
            WHERE text LIKE ?
            ORDER BY id DESC LIMIT ?;
            """, (like, limit))
        rows = [Turn(r["role"], r["text"], r["ts"]) for r in c.fetchall()]
        rows.reverse()
        return rows

    def count(self) -> int:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) AS n FROM turns")
        return int(c.fetchone()["n"])

    def close(self):
        self.conn.close()
