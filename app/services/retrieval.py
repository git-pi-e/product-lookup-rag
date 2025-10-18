import json
from typing import Any, Dict, List

import numpy as np
from neo4j.exceptions import Neo4jError

from ..core.entities import entity_relationship_match
from ..graph import get_graph
import logging

logger = logging.getLogger(__name__)


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
        try:
            logger.debug("Attempting server-side GDS similarity query")
            results = self.graph.query(query, params=params)
            logger.debug(
                "Server query returned %d rows", len(results) if results else 0
            )
            # annotate which method returned results
            for r in results:
                if isinstance(r, dict) and "p" in r:
                    r.setdefault("_meta", {})[
                        "retrieval_method"
                    ] = "entities_server_gds"
            return results
        except Neo4jError as e:
            msg = str(e)
            logger.warning("Server GDS query failed: %s", msg)
            if (
                "gds.similarity" in msg
                or "Unknown function 'gds.similarity.cosine'" in msg
            ):
                logger.info(
                    "Falling back to client-side embedding similarity computation"
                )
                # Fallback: fetch candidate products and entity embeddings, compute cosine in Python
                keys = [k for k in query_data.keys() if k != "product"]
                # Build OPTIONAL MATCH clauses and collect embeddings per key
                optional_matches = []
                collect_returns = ["p.id AS id", "p.name AS name"]
                for key in keys:
                    rel = entity_relationship_match[key]
                    var = f"{key}Var"
                    optional_matches.append(
                        f"OPTIONAL MATCH (p)-[:{rel}]->({var}:{key})"
                    )
                    collect_returns.append(
                        f"collect(DISTINCT {var}.embedding) AS emb_{key}"
                    )

                cypher = (
                    "MATCH (p:Product)\n"
                    + "\n".join(optional_matches)
                    + "\nRETURN "
                    + ", ".join(collect_returns)
                )
                rows = self.graph.query(cypher)

                def cosine(a, b):
                    a = np.array(a, dtype=float)
                    b = np.array(b, dtype=float)
                    if a.size == 0 or b.size == 0:
                        return 0.0
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    if denom == 0:
                        return 0.0
                    return float(np.dot(a, b) / denom)

                results = []
                threshold = 0.81
                for r in rows:
                    match_all = True
                    for key in keys:
                        query_emb = params.get(f"{key}Embedding")
                        candidate_embs = r.get(f"emb_{key}") or []
                        # compute max similarity among candidate entity embeddings
                        max_sim = 0.0
                        for ce in candidate_embs:
                            if ce:
                                try:
                                    sim = cosine(ce, query_emb)
                                except Exception:
                                    sim = 0.0
                                if sim > max_sim:
                                    max_sim = sim
                        if max_sim < threshold:
                            match_all = False
                            break
                    if match_all:
                        results.append(
                            {"p": {"id": r.get("id"), "name": r.get("name")}}
                        )
                logger.debug(
                    "Client-side embedding fallback produced %d results", len(results)
                )
                for r in results:
                    if isinstance(r, dict) and "p" in r:
                        r.setdefault("_meta", {})[
                            "retrieval_method"
                        ] = "entities_client_fallback"
                return results
            raise

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
        try:
            results = self.graph.query(
                query, params={"embedding": embedding, "threshold": threshold}
            )
            for r in results:
                if isinstance(r, dict) and "p" in r:
                    r.setdefault("_meta", {})["retrieval_method"] = "prompt_server_gds"
            return results
        except Neo4jError as e:
            msg = str(e)
            if (
                "gds.similarity" in msg
                or "Unknown function 'gds.similarity.cosine'" in msg
            ):
                # Fallback: fetch p.embedding and compute cosine in Python
                rows = self.graph.query(
                    "MATCH (p:Product) WHERE p.embedding IS NOT NULL RETURN p.id AS id, p.name AS name, p.embedding AS embedding"
                )

                def cosine(a, b):
                    a = np.array(a, dtype=float)
                    b = np.array(b, dtype=float)
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    if denom == 0:
                        return 0.0
                    return float(np.dot(a, b) / denom)

                results = []
                for r in rows:
                    emb = r.get("embedding")
                    if emb:
                        sim = cosine(emb, embedding)
                        if sim > threshold:
                            candidate = {
                                "p": {"id": r.get("id"), "name": r.get("name")}
                            }
                            candidate.setdefault("_meta", {})[
                                "retrieval_method"
                            ] = "prompt_client_fallback"
                            results.append(candidate)
                return results
            raise

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
