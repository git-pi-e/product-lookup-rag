from functools import lru_cache
from typing import Optional

try:
    from langchain_neo4j import Neo4jGraph  # Preferred in newer LangChain
except ImportError:  # Fallback for environments without langchain-neo4j
    from langchain_community.graphs import Neo4jGraph

from neo4j.exceptions import ClientError

from .config import get_settings


@lru_cache(maxsize=1)
def get_graph() -> Neo4jGraph:
    settings = get_settings()
    # Try with provided database first; on failure, fallback to common defaults
    for db_name in [settings.neo4j_database, "neo4j", None]:
        try:
            graph = Neo4jGraph(
                url=settings.neo4j_url,
                username=settings.neo4j_user,
                password=settings.neo4j_password,
                database=db_name,  # type: ignore[arg-type]
            )
            # Touch the schema to validate connection/db selection
            if hasattr(graph, "refresh_schema"):
                try:
                    graph.refresh_schema()  # type: ignore[attr-defined]
                except Exception:
                    # If schema refresh fails, try next db option
                    raise
            return graph
        except ClientError:
            continue
        except Exception:
            continue
    # If all attempts fail, raise a clear error
    raise RuntimeError(
        "Failed to connect to Neo4j. Check NEO4J_URL/USER/PASSWORD and NEO4J_DATABASE (try leaving it blank or 'neo4j')."
    )
