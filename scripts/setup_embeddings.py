import os
import sys
import hashlib
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.graph import get_graph


from langchain_community.vectorstores.neo4j_vector import Neo4jVector

from openai import OpenAI

# Try to import LangChain OpenAI embeddings wrapper if available; otherwise operate
# using the OpenAI client directly for embedding generation and skip vector index creation.
try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:
    try:
        from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore
    except Exception:
        OpenAIEmbeddings = None  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="Create embeddings and Neo4j vector indexes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to DB or call OpenAI; just report actions",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    load_dotenv()
    settings = get_settings()
    url = settings.neo4j_url
    username = settings.neo4j_user
    password = settings.neo4j_password

    graph = get_graph()

    def ensure_index(
        index_name: str, label: str, text_props: List[str], embedding_prop: str
    ):
        # Strict mode: use only the configured NEO4J_DATABASE from settings. Fail fast on mismatch.
        idx_exists = graph.query(
            """
            SHOW INDEXES YIELD name
            WHERE name = $name
            RETURN count(*) AS c
            """,
            params={"name": index_name},
        )
        count = idx_exists[0]["c"] if idx_exists else 0
        if count == 0:
            if OpenAIEmbeddings is None or Neo4jVector is None:
                raise RuntimeError(
                    f"Required packages missing: OpenAIEmbeddings or Neo4jVector. Install langchain-openai and langchain-community or use the OpenAI client fallback."
                )

            # Use only the explicitly configured database; do not attempt fallbacks.
            configured_db = settings.neo4j_database
            if not configured_db:
                raise RuntimeError(
                    "NEO4J_DATABASE is not set; set it in .env and retry."
                )

            logging.info(
                "Index '%s' missing for label %s. %s",
                index_name,
                label,
                "(dry-run)" if args.dry_run else "",
            )
            if args.dry_run:
                return

            # Attempt to create the index targeting the configured database.
            try:
                Neo4jVector.from_existing_graph(
                    OpenAIEmbeddings(model=settings.embeddings_model),
                    url=url,
                    username=username,
                    password=password,
                    index_name=index_name,
                    node_label=label,
                    text_node_properties=text_props,
                    embedding_node_property=embedding_prop,
                    database=configured_db,
                )
            except TypeError:
                # Some versions might not accept `database`; allow a single retry without it but still
                # only operate if the configured DB matches the graph's active DB.
                Neo4jVector.from_existing_graph(
                    OpenAIEmbeddings(model=settings.embeddings_model),
                    url=url,
                    username=username,
                    password=password,
                    index_name=index_name,
                    node_label=label,
                    text_node_properties=text_props,
                    embedding_node_property=embedding_prop,
                )

    # Schema expectations
    REQUIRED_PRODUCT_PROPS = ["id", "name", "title"]
    OPTIONAL_PRODUCT_PROPS = ["embedding", "embedding_hash"]
    EXPECTED_PRODUCT_PROPS = REQUIRED_PRODUCT_PROPS + OPTIONAL_PRODUCT_PROPS
    EXPECTED_ENTITY_PROPS = ["value", "embedding", "embedding_hash"]

    # Create product vector index once if missing
    ensure_index("products", "Product", ["name", "title"], "embedding")

    # Discover entity labels from data file if present; else query db labels
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "amazon_product_kg.json"
    )
    entity_labels = set()
    try:
        df = pd.read_json(data_path)
        entity_labels = set(df["entity_type"].unique())
    except Exception:
        labels = graph.query("CALL db.labels()")
        entity_labels = set(
            l["label"] if "label" in l else list(l.values())[0] for l in labels
        )
    entity_labels.discard("Product")

    # Report existing schema vs expected to help users align ingestion/schema
    def report_schema():
        try:
            prop_rows = graph.query(
                "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"
            )
            existing_props = {
                r.get("propertyKey") or list(r.values())[0] for r in prop_rows
            }
        except Exception:
            existing_props = set()

        print("\nSchema Report:\n----------------")
        print("Existing property keys:", sorted(existing_props))

        missing_product = [p for p in EXPECTED_PRODUCT_PROPS if p not in existing_props]
        if missing_product:
            print("Missing expected Product properties:", missing_product)
            print(
                "Tip: run scripts/ingest.py (targeting the correct NEO4J_DATABASE) to populate these properties."
            )
        else:
            print("All expected Product properties present.")

        # Check some sample nodes for each entity label
        for label in sorted(entity_labels):
            try:
                sample = graph.query(
                    f"MATCH (n:{label}) RETURN keys(n) AS keys LIMIT 3"
                )
                keys = set()
                for r in sample:
                    k = (
                        r.get("keys")
                        if isinstance(r.get("keys"), list)
                        else list(r.values())[0]
                    )
                    if isinstance(k, (list, tuple, set)):
                        keys.update(k)
                    elif k is not None:
                        keys.add(str(k))
                missing = [p for p in EXPECTED_ENTITY_PROPS if p not in keys]
                if missing:
                    print(f"Label {label}: missing expected props: {missing}")
                else:
                    print(f"Label {label}: expected props present")
            except Exception:
                print(f"Label {label}: could not sample nodes (maybe none exist)")

    report_schema()

    # Ensure entity indexes exist once
    for label in entity_labels:
        ensure_index(label, label, ["value"], "embedding")

    # Idempotent re-embedding using content hashing
    # Fail fast: ensure required Product properties exist
    prop_rows = graph.query(
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"
    )
    existing_props = {r.get("propertyKey") or list(r.values())[0] for r in prop_rows}
    missing_required = [p for p in REQUIRED_PRODUCT_PROPS if p not in existing_props]
    if missing_required:
        raise RuntimeError(
            f"Missing required Product properties in DB ({missing_required}). Run scripts/ingest.py to populate required fields and retry."
        )

    missing_optional = [p for p in OPTIONAL_PRODUCT_PROPS if p not in existing_props]
    if missing_optional:
        print(
            f"Note: optional embedding properties missing (will be created): {missing_optional}"
        )

    client = None if args.dry_run else OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    def sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_products(batch_size: int = 500):
        skip = 0
        while True:
            rows = graph.query(
                """
                MATCH (p:Product)
                // Try to return canonical properties; fall back to internal id when missing
                RETURN coalesce(p.id, id(p)) AS id, p.name AS name, p.title AS title, p.embedding_hash AS h
                SKIP $skip LIMIT $limit
                """,
                params={"skip": skip, "limit": batch_size},
            )
            if not rows:
                break
            updates: List[Dict] = []
            texts: List[str] = []
            ids: List[int] = []
            hashes: List[str] = []
            for r in rows:
                name = r.get("name") or ""
                title = r.get("title") or ""
                text = f"{name} {title}".strip()
                h = sha256(text)
                if not r.get("h") or r.get("h") != h:
                    texts.append(text)
                    ids.append(r["id"])
                    hashes.append(h)
            if texts:
                if args.dry_run:
                    logging.info(
                        "DRY RUN: would embed %d product texts (sample first 3): %s",
                        len(texts),
                        texts[:3],
                    )
                else:
                    assert client is not None
                    resp = client.embeddings.create(
                        model=settings.embeddings_model,
                        input=texts,
                    )
                    vectors = [d.embedding for d in resp.data]
                    updates = [
                        {"id": i, "embedding": v, "hash": h}
                        for i, v, h in zip(ids, vectors, hashes)
                    ]
                    graph.query(
                        """
                        UNWIND $updates AS row
                        MATCH (p:Product)
                        WHERE coalesce(p.id, id(p)) = row.id
                        SET p.embedding = row.embedding, p.embedding_hash = row.hash
                        """,
                        params={"updates": updates},
                    )
            skip += batch_size

    def embed_entities(label: str, batch_size: int = 500):
        skip = 0
        while True:
            rows = graph.query(
                f"""
                MATCH (e:{label})
                RETURN e.value AS value, e.embedding_hash AS h
                SKIP $skip LIMIT $limit
                """,
                params={"skip": skip, "limit": batch_size},
            )
            if not rows:
                break
            texts: List[str] = []
            values: List[str] = []
            hashes: List[str] = []
            for r in rows:
                val = r.get("value") or ""
                h = sha256(val)
                if not r.get("h") or r.get("h") != h:
                    texts.append(val)
                    values.append(val)
                    hashes.append(h)
            if texts:
                if args.dry_run:
                    logging.info(
                        "DRY RUN: would embed %d entity texts for label %s (sample first 3): %s",
                        len(texts),
                        label,
                        texts[:3],
                    )
                else:
                    assert client is not None
                    resp = client.embeddings.create(
                        model=settings.embeddings_model,
                        input=texts,
                    )
                    vectors = [d.embedding for d in resp.data]
                    updates = [
                        {"value": v, "embedding": vec, "hash": h}
                        for v, vec, h in zip(values, vectors, hashes)
                    ]
                    graph.query(
                        f"""
                        UNWIND $updates AS row
                        MATCH (e:{label} {{value: row.value}})
                        SET e.embedding = row.embedding, e.embedding_hash = row.hash
                        """,
                        params={"updates": updates},
                    )
            skip += batch_size

    # Execute idempotent embedding updates
    embed_products()
    for label in sorted(entity_labels):
        embed_entities(label)


if __name__ == "__main__":
    main()
