import json
from typing import Any, Dict, List

from ..core.entities import entity_relationship_match
from ..graph import get_graph


class RetrievalService:
    def __init__(self, embedder):
        self.graph = get_graph()
        self.embedder = embedder

    def build_cypher(self, extracted_json: str, threshold: float = 0.81) -> str:
        query_data = json.loads(extracted_json)

        embeddings_data: List[str] = []
        for key in query_data.keys():
            if key != "product":
                embeddings_data.append(f"${key}Embedding AS {key}Embedding")
        query = "WITH " + ",\n".join(embeddings_data) if embeddings_data else ""

        query += "\nMATCH (p:Product)\n"

        match_data: List[str] = []
        for key in query_data.keys():
            if key != "product":
                relationship = entity_relationship_match[key]
                match_data.append(f"(p)-[:{relationship}]->({key}Var:{key})")
        if match_data:
            query += "MATCH " + ",\n".join(match_data)

        similarity_data: List[str] = []
        for key in query_data.keys():
            if key != "product":
                similarity_data.append(
                    f"gds.similarity.cosine({key}Var.embedding, ${key}Embedding) > {threshold}"
                )
        if similarity_data:
            query += "\nWHERE " + " AND ".join(similarity_data)

        query += "\nRETURN p"
        return query

    def query_by_entities(self, extracted_json: str) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        query_data = json.loads(extracted_json)
        for key, val in query_data.items():
            if key != "product":
                params[f"{key}Embedding"] = self.embedder(str(val))
        query = self.build_cypher(extracted_json)
        return self.graph.query(query, params=params)

    def query_by_prompt_similarity(
        self, prompt: str, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        embedding = self.embedder(prompt)
        query = """
            WITH $embedding AS inputEmbedding
            MATCH (p:Product)
            WHERE gds.similarity.cosine(inputEmbedding, p.embedding) > $threshold
            RETURN p
        """
        return self.graph.query(
            query, params={"embedding": embedding, "threshold": threshold}
        )

    def similar_items(
        self, product_id: int, relationships_threshold: int = 3
    ) -> List[Dict[str, Any]]:
        similar_items: List[Dict[str, Any]] = []
        query_category = """
            MATCH (p:Product {id: $product_id})-[:hasCategory]->(c:category)
            MATCH (p)-->(entity)
            WHERE NOT entity:category
            MATCH (n:Product)-[:hasCategory]->(c)
            MATCH (n)-->(commonEntity)
            WHERE commonEntity = entity AND p.id <> n.id
            RETURN DISTINCT n;
        """
        result_category = self.graph.query(
            query_category, params={"product_id": int(product_id)}
        )

        query_common_entities = """
            MATCH (p:Product {id: $product_id})-->(entity),
                  (n:Product)-->(entity)
            WHERE p.id <> n.id
            WITH n, COUNT(DISTINCT entity) AS commonEntities
            WHERE commonEntities >= $threshold
            RETURN n;
        """
        result_common = self.graph.query(
            query_common_entities,
            params={
                "product_id": int(product_id),
                "threshold": relationships_threshold,
            },
        )

        for row in result_category:
            similar_items.append({"id": row["n"]["id"], "name": row["n"]["name"]})
        for row in result_common:
            rid = row["n"]["id"]
            if not any(item["id"] == rid for item in similar_items):
                similar_items.append({"id": rid, "name": row["n"]["name"]})
        return similar_items
