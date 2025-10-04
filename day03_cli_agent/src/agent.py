# agent.py
"""
Day 3: CLI Assistant Agent
--------------------------
Now the agent can:
- Take prompts directly from the command line
- Be used as a small tool for quick queries
- Keep the Day 2 structured behavior (AI coding tutor)
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
import sys
from openai import OpenAI

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Combine all command-line args into one prompt
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("You: ")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an AI coding tutor. "
                "Explain answers in concise bullet points under 100 words."
            )},
            {"role": "user", "content": user_input}
        ]
    )

    print("Agent:", response.choices[0].message.content.strip())


if __name__ == "__main__":
    main()
