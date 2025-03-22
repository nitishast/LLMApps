import os
from dotenv import load_dotenv

load_dotenv()

# MODEL = "gemini-1.5-pro"
MODEL = 'Gemini 1.5 Flash'
CHAT_MODEL = "gemini-1.5-pro-exp-0801"
EMBEDDINGS_MODEL = "models/embeddings-001"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME=os.getenv("OPENAI_MODEL_NAME")
OPENAI_API_BASE=os.getenv("OPENAI_API_BASE")


