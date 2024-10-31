from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI

def load_api_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

api_key = load_api_key()
print("Testing OpenAI initialization with API key:", api_key)

try:
    llm = OpenAI(openai_api_key=api_key)
    print("OpenAI initialized successfully.")
except Exception as e:
    print("Error initializing OpenAI:", e)
