# src/agent.py
"""
Day 4: Calculator Agent
-----------------------
- Detect math queries
- Use safe_eval tool for math
- Fall back to LLM for everything else
"""
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
import sys
from openai import OpenAI
from tools import safe_eval, looks_like_math

MODEL = "gpt-4o-mini"  # fast & inexpensive

def llm_reply(client, user_input: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "You are an AI coding tutor. "
                "Be concise, use short bullet points when useful. "
                "If user asks for math result explicitly, defer to the calculator tool result provided."
            )},
            {"role": "user", "content": user_input}
        ],
    )
    return resp.choices[0].message.content.strip()

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # prompt from CLI args or interactive
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("You: ")

    if looks_like_math(user_input):
        try:
            value = safe_eval(user_input)
            print(f"Agent (calculator): {value}")
            return
        except ZeroDivisionError:
            print("Agent (calculator): Division by zero is undefined.")
            return
        except ValueError:
            # fall through to LLM if math parsing fails
            pass

    print("Agent:", llm_reply(client, user_input))

if __name__ == "__main__":
    main()
