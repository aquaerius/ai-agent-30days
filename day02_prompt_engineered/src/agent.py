# agent.py
"""
Day 2: Prompt-Engineered Agent
-------------------------------
Adds structured prompts:
- Role: The agent takes on a specific persona
- Style: Responses are concise and structured
- Constraints: Agent sticks to rules
"""
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
from openai import OpenAI

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    user_input = input("You: ")

    # Define system + user messages for better control
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an AI coding tutor. "
                "Always explain concepts in clear, short bullet points. "
                "Keep answers under 100 words."
            )},
            {"role": "user", "content": user_input}
        ]
    )

    print("Agent:", response.choices[0].message.content.strip())


if __name__ == "__main__":
    main()
