import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "jason_documents")

# Scraping Configuration
MEDIUM_USERNAME = os.getenv("MEDIUM_USERNAME")
LINKEDIN_PROFILE_URL = os.getenv("LINKEDIN_PROFILE_URL")

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
