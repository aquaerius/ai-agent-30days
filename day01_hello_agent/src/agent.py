# agent.py
"""
Day 1: Hello Agent
-------------------
This is the simplest AI agent possible: 
- Takes user input from the command line
- Sends it to the OpenAI API
- Prints the model's response
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
from openai import OpenAI

def main():
    # Load API key from environment variable
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Ask user for input
    user_input = input("You: ")

    # Call OpenAI model
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # small + fast model
        messages=[{"role": "user", "content": user_input}]
    )

    # Print response
    print("Agent:", response.choices[0].message.content.strip())


if __name__ == "__main__":
    main()
