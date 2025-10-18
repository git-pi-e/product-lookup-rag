from functools import lru_cache
from typing import Optional

from langchain_community.graphs import Neo4jGraph

from .config import get_settings


@lru_cache(maxsize=1)
def get_graph() -> Neo4jGraph:
    settings = get_settings()
    graph = Neo4jGraph(
        url=settings.neo4j_url,
        username=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,  # type: ignore[arg-type]
    )
    return graph
