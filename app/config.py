import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel


# Load variables from a local .env file if present
load_dotenv()


class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    neo4j_url: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: Optional[str] = os.getenv("NEO4J_DATABASE") or None
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    # Adjusted defaults for similarity threshold/top_k for testing
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.36"))
    similarity_top_k: int = int(os.getenv("SIMILARITY_TOP_K", "25"))


def get_settings() -> Settings:
    return Settings()
