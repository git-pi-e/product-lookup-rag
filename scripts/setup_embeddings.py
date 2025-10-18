import os
import sys
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.graph import get_graph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings


def main():
    load_dotenv()
    settings = get_settings()
    url = settings.neo4j_url
    username = settings.neo4j_user
    password = settings.neo4j_password

    # Create product vector index from existing graph
    Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(model=settings.embeddings_model),
        url=url,
        username=username,
        password=password,
        index_name="products",
        node_label="Product",
        text_node_properties=["name", "title"],
        embedding_node_property="embedding",
    )

    # Discover entity labels from data file if present; else query graph
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "amazon_product_kg.json"
    )
    entity_labels = set()
    try:
        df = pd.read_json(data_path)
        entity_labels = set(df["entity_type"].unique())
    except Exception:
        # Fallback: try to infer entity labels from the graph schema
        graph = get_graph()
        schema = graph.query("CALL db.schema.visualization()")
        # Best-effort: collect labels other than Product
        for row in schema:
            # row structure depends on Neo4j version; skipping robust parsing here
            pass

    for label in entity_labels:
        Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(model=settings.embeddings_model),
            url=url,
            username=username,
            password=password,
            index_name=label,
            node_label=label,
            text_node_properties=["value"],
            embedding_node_property="embedding",
        )


if __name__ == "__main__":
    main()
