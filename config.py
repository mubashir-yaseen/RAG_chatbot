"""
Configuration settings for the RAG system.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", 
    "sentence-transformers/all-MiniLM-L6-v2"
)

# RAG Configuration
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
K_RESULTS = int(os.getenv("RAG_K_RESULTS", "3"))
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.3"))

# Vector Store Configuration
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_stores")
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "faiss")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# Create necessary directories
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


def validate_config():
    """Validate that all required configuration is set."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it or provide it in the Streamlit UI."
        )
    return True


if __name__ == "__main__":
    print("RAG System Configuration:")
    print(f"  OpenAI Model: {OPENAI_MODEL}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"  K Results: {K_RESULTS}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Vector Store Dir: {VECTOR_STORE_DIR}")
