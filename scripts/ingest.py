import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import logging
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph import get_graph


def sanitize(text):
    text = str(text).replace("'", "").replace('"', "").replace("{", "").replace("}", "")
    return text


def main():
    parser = argparse.ArgumentParser(description="Ingest product KG into Neo4j")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't write to DB; just print actions"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    load_dotenv()
    data_path = os.getenv(
        "DATA_JSON",
        str(Path(__file__).resolve().parents[1] / "data" / "amazon_product_kg.json"),
    )
    with open(data_path, "r") as f:
        json_data = json.load(f)

    graph = get_graph()
    i = 1
    for obj in json_data:
        logging.debug(
            "%d. %s -%s-> %s",
            i,
            obj.get("product_id"),
            obj.get("relationship"),
            obj.get("entity_value"),
        )
        i += 1
        # Ensure deterministic schema: always set id (string), name, title, bullet_points, size
        product_id = sanitize(obj["product_id"])
        product_name = sanitize(obj.get("product") or obj.get("product_name") or "")
        product_title = sanitize(obj.get("TITLE") or "")
        bullet_points = sanitize(obj.get("BULLET_POINTS") or "")
        size = sanitize(obj.get("PRODUCT_LENGTH") or "null")

        # Use parameterized Cypher to avoid injection and ensure proper quoting
        query = f"""
            MERGE (product:Product {{id: $pid}})
            ON CREATE SET product.name = $name,
                          product.title = $title,
                          product.bullet_points = $bullets,
                          product.size = $size

            MERGE (entity:{obj['entity_type']} {{value: $evalue}})

            MERGE (product)-[:{obj['relationship']}]->(entity)
        """
        params = {
            "pid": product_id,
            "name": product_name,
            "title": product_title,
            "bullets": bullet_points,
            "size": None if size == "null" else size,
            "evalue": sanitize(obj["entity_value"]),
        }
        if args.dry_run:
            logging.info("DRY RUN: would execute Cypher with params: %s", params)
        else:
            logging.debug("Executing Cypher MERGE for product %s", product_id)
            graph.query(query, params=params)

    # After ingest, ensure all Product nodes have a stable id property (elementId) if missing
    if args.dry_run:
        logging.info("DRY RUN: would set missing p.id = elementId(p)")
    else:
        updated = graph.query(
            """
            MATCH (p:Product)
            WHERE p.id IS NULL
            SET p.id = elementId(p)
            RETURN count(p) AS updated_count
            """
        )
        logging.info("Updated missing product ids: %s", updated)


if __name__ == "__main__":
    main()
