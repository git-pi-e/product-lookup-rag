import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph import get_graph


def sanitize(text):
    text = str(text).replace("'", "").replace('"', "").replace("{", "").replace("}", "")
    return text


def main():
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
        print(
            f"{i}. {obj['product_id']} -{obj['relationship']}-> {obj['entity_value']}"
        )
        i += 1
        query = f"""
            MERGE (product:Product {{id: {obj['product_id']}}})
            ON CREATE SET product.name = "{sanitize(obj['product'])}",
                           product.title = "{sanitize(obj['TITLE'])}",
                           product.bullet_points = "{sanitize(obj['BULLET_POINTS'])}",
                           product.size = {sanitize(obj['PRODUCT_LENGTH'])}

            MERGE (entity:{obj['entity_type']} {{value: "{sanitize(obj['entity_value'])}"}})

            MERGE (product)-[:{obj['relationship']}]->(entity)
        """
        graph.query(query)


if __name__ == "__main__":
    main()
