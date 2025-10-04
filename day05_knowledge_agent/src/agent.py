# src/agent.py
"""
Day 5: Knowledge Agent (Local RAG with TF-IDF)
- Loads a local index built from PDFs/TXTs in ./data
- Retrieves top-k chunks for a query
- Sends them as context to the LLM
- Produces a grounded answer with inline citations [doc:chunk]
"""
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env
import os, sys
from typing import List
from openai import OpenAI
from retriever import load_index, retrieve

MODEL = "gpt-4o-mini"
MAX_CONTEXT_CHARS = 2200  # keep prompt small/cheap

def format_context(retrieved) -> str:
    # retrieved: List[(Chunk, score)]
    blocks: List[str] = []
    used = 0
    for chunk, score in retrieved:
        tag = f"[{chunk.doc_path}#{chunk.chunk_id}]"
        text = chunk.text.strip()
        # pack within a rough budget
        piece = f"{tag}\n{text}\n"
        if used + len(piece) > MAX_CONTEXT_CHARS and blocks:
            break
        blocks.append(piece)
        used += len(piece)
    return "\n---\n".join(blocks)

def answer_with_rag(client: OpenAI, question: str, context_block: str) -> str:
    sys_prompt = (
        "You are a precise assistant that answers ONLY using the provided context. "
        "If the answer is not contained within the context, say: 'I don't know based on the provided documents.' "
        "Cite sources inline using their tags like [doc#chunk]. Be concise."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer (with citations):"}
    ]
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content.strip()

def main():
    base = os.path.dirname(os.path.dirname(__file__))
    index_path = os.path.join(base, "tfidf.index.pkl")
    if not os.path.exists(index_path):
        print("Index not found. Run: python src/ingest.py")
        sys.exit(1)

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Ask a question about your docs: ")

    idx = load_index(index_path)
    top = retrieve(idx, question, k=5)
    if not top:
        print("No relevant passages found. Add docs to ./data and re-index.")
        sys.exit(0)

    context_block = format_context(top)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    answer = answer_with_rag(client, question, context_block)
    print("\n=== Retrieved Context ===")
    print(context_block)
    print("\n=== Answer ===")
    print(answer)

if __name__ == "__main__":
    main()
