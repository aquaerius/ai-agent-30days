# src/agent.py
"""
Day 6: Persistent Memory Agent
- Loads recent + relevant memory from SQLite
- Uses that context to answer
- Saves the new turn back to memory
"""

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env
import os
import sys
from typing import List
from openai import OpenAI
from memory import MemoryStore, Turn

MODEL = "gpt-4o-mini"
RECENT_K = 6           # include last K turns
KEYWORD_LIMIT = 4      # include up to 4 keyword hits

def format_memory_block(recent: List[Turn], keywords: List[Turn]) -> str:
    def fmt(ts: str, role: str, text: str) -> str:
        return f"[{ts}][{role}] {text}"
    blocks = []
    if recent:
        blocks.append("Recent context:\n" + "\n".join(fmt(t.ts, t.role, t.text) for t in recent))
    if keywords:
        blocks.append("Keyword matches:\n" + "\n".join(fmt(t.ts, t.role, t.text) for t in keywords))
    return "\n\n".join(blocks) if blocks else "(no prior context)"

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    mem = MemoryStore()  # memory.db in project root

    # Input prompt via CLI or interactive
    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("You: ")

    # Simple keyword heuristic: if the user adds tags like #topic, search those
    tags = [w[1:] for w in user_input.split() if w.startswith("#")]
    keywords = []
    for tag in tags:
        keywords.extend(mem.recall_keywords(tag, limit=KEYWORD_LIMIT))

    recent = mem.recall_recent(RECENT_K)
    memory_block = format_memory_block(recent, keywords)

    sys_prompt = (
        "You are an assistant that uses provided memory to stay consistent with the user's history. "
        "Prefer information from memory when relevant, but do not fabricate facts. "
        "Be concise and helpful."
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Memory:\n{memory_block}\n\nCurrent request:\n{user_input}"}
    ]

    resp = client.chat.completions.create(model=MODEL, messages=messages)
    agent_text = resp.choices[0].message.content.strip()

    print("Agent:", agent_text)

    # Save to memory
    mem.add_interaction(user_input, agent_text)
    mem.close()

if __name__ == "__main__":
    main()
