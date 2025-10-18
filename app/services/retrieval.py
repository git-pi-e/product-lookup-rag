import json
from typing import Any, Dict, List

import numpy as np
from neo4j.exceptions import Neo4jError

from ..core.entities import entity_relationship_match
from ..graph import get_graph
import logging
from typing import Optional
from ..config import get_settings

logger = logging.getLogger(__name__)
# Default to INFO for this module; keep debug calls available for troubleshooting
logger.setLevel(logging.INFO)


class RetrievalService:
    def __init__(self, embedder):
        self.graph = get_graph()
        self.embedder = embedder
        # Detect available server-side similarity function (GDS / vector functions)
        self.gds_fn: str | None = None
        try:
            self.gds_fn = self._detect_gds_function()
        except Exception:
            self.gds_fn = None
        # Load similarity settings and inspect embedding schema for diagnostics
        settings = get_settings()
        self.similarity_threshold = settings.similarity_threshold
        self.similarity_top_k = settings.similarity_top_k

        # Inspect embedding schema to avoid running broken server-side queries
        self.embedding_dim: Optional[int] = None
        try:
            self._inspect_product_embeddings()
        except Exception:
            logger.debug(
                "Failed to inspect product embeddings at startup", exc_info=True
            )

    def _detect_gds_function(self) -> str | None:
        # probing for common GDS/vector cosine function names
        candidates = [
            "gds.similarity.cosine",
            "gds.alpha.similarity.cosine",
            "vector.similarity.cosine",
        ]
        for fn in candidates:
            try:
                test_q = f"RETURN {fn}([0.1,0.2],[0.1,0.2]) AS sim"
                try:
                    r = self.graph.query(test_q)
                except Exception:
                    # call may fail if function exists but call context differs
                    r = None
                if r:
                    logger.info("Detected server-side similarity function: %s", fn)
                    return fn
            except Exception:
                continue
        logger.info(
            "No server-side similarity function detected; will use client-side fallbacks"
        )
        return None

    def _inspect_product_embeddings(self) -> None:
        """Fetch a small sample of product embeddings to validate presence, type and dimension.

        Sets self.embedding_dim when a valid vector is found. Logs warnings when issues are
        detected so callers can choose safe fallbacks.
        """
        try:
            sample = self.graph.query(
                "MATCH (p:Product) WHERE p.embedding IS NOT NULL RETURN p.embedding AS e LIMIT 3"
            )
        except Exception:
            logger.debug(
                "Could not query product embeddings for inspection", exc_info=True
            )
            return

        if not sample:
            logger.warning(
                "No product embeddings found in DB (p.embedding is NULL for all sampled nodes)"
            )
            return

        for row in sample:
            emb = None
            if isinstance(row, dict):
                emb = row.get("e") or row.get("embedding")
            else:
                # support different result shapes
                try:
                    emb = list(row.values())[0]
                except Exception:
                    emb = None

            if emb is None:
                logger.warning("Found a product row without embedding")
                continue

            # Validate numeric vector
            try:
                arr = np.array(emb, dtype=float)
            except Exception:
                logger.warning(
                    "Product embedding is not a numeric vector (sample: %s)",
                    str(emb)[:200],
                )
                continue

            if arr.ndim != 1:
                logger.warning(
                    "Product embedding has unexpected shape (ndim=%d)", arr.ndim
                )
                continue

            # Record embedding dimension and stop
            self.embedding_dim = int(arr.shape[0])
            logger.info("Detected product embedding dimension: %d", self.embedding_dim)
            return

        logger.warning(
            "No valid numeric product embeddings detected in sample; server-side similarity may fail"
        )

    def build_cypher(self, extracted_json: str, threshold: float = 0.81) -> str:
        query_data = json.loads(extracted_json)
        # If a server-side similarity function exists, build the embedding-based WHERE query
        if self.gds_fn:
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
                        f"{self.gds_fn}({key}Var.embedding, ${key}Embedding) > {threshold}"
                    )
            if similarity_data:
                query += "\nWHERE " + " AND ".join(similarity_data)
            query += "\nRETURN p"
            return query

        # No server-side similarity: fallback to deterministic exact-match by entity values
        query = "MATCH (p:Product)\n"
        where_clauses: List[str] = []
        for key in query_data.keys():
            if key != "product":
                relationship = entity_relationship_match[key]
                # Use case-insensitive match on entity.value
                query += f"MATCH (p)-[:{relationship}]->({key}Var:{key})\n"
                where_clauses.append(f"toLower({key}Var.value) = toLower(${key}Value)")
        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses)
        query += "\nRETURN p"
        return query

    def query_by_entities(self, extracted_json: str) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        query_data = json.loads(extracted_json)
        # Prepare params depending on whether server-side similarity is used
        if self.gds_fn:
            for key, val in query_data.items():
                if key != "product":
                    params[f"{key}Embedding"] = self.embedder(str(val))
        else:
            for key, val in query_data.items():
                if key != "product":
                    params[f"{key}Value"] = str(val)
        query = self.build_cypher(extracted_json)
        # First try a deterministic exact-match on entity values (not embeddings).
        try:
            exact_keys = [k for k in query_data.keys() if k != "product"]
            if exact_keys:
                exact_cypher = "MATCH (p:Product)\n"
                exact_clauses: List[str] = []
                for key in exact_keys:
                    rel = entity_relationship_match[key]
                    exact_cypher += f"MATCH (p)-[:{rel}]->({key}Var:{key})\n"
                    exact_clauses.append(
                        f"toLower({key}Var.value) = toLower(${key}Value)"
                    )
                if exact_clauses:
                    exact_cypher += "WHERE " + " AND ".join(exact_clauses)
                exact_cypher += "\nRETURN p"
                exact_params = {
                    f"{k}Value": str(v) for k, v in query_data.items() if k != "product"
                }
                logger.debug(
                    "Attempting deterministic exact-match query: %s", exact_cypher
                )
                logger.debug("Exact params: %s", exact_params)
                exact_rows = self.graph.query(exact_cypher, params=exact_params)
                if exact_rows:
                    logger.info("Exact-match query returned %d rows", len(exact_rows))
                    for r in exact_rows:
                        if isinstance(r, dict) and "p" in r:
                            r.setdefault("_meta", {})[
                                "retrieval_method"
                            ] = "entities_exact_match"
                    return exact_rows
        except Exception:
            logger.debug(
                "Deterministic exact-match check failed, continuing to similarity logic",
                exc_info=True,
            )
        try:
            logger.debug("Attempting server-side GDS similarity query")
            logger.debug("Cypher: %s", query)
            logger.debug("Params: %s", params)
            results = self.graph.query(query, params=params)
            logger.info("Server query returned %d rows", len(results) if results else 0)
            # annotate which method returned results
            for r in results:
                if isinstance(r, dict) and "p" in r:
                    r.setdefault("_meta", {})[
                        "retrieval_method"
                    ] = "entities_server_gds"
            # If server returned no rows, fall back to client-side product embedding similarity
            if not results:
                logger.info(
                    "Server-side entity similarity returned no rows; falling back to client-side product embedding comparison"
                )
                # reuse the client-side flow from the exception handler
                keys = [k for k in query_data.keys() if k != "product"]
                if not keys:
                    return []
                combined_text = " ".join([f"{k}: {query_data[k]}" for k in keys])
                try:
                    query_emb = self.embedder(combined_text)
                except Exception:
                    logger.exception(
                        "Failed to compute embedding for combined entity text"
                    )
                    return []

                rows = self.graph.query(
                    "MATCH (p:Product) WHERE p.embedding IS NOT NULL RETURN p.id AS id, p.name AS name, p.embedding AS embedding"
                )

                def cosine(a, b):
                    try:
                        a = np.array(a, dtype=float)
                        b = np.array(b, dtype=float)
                    except Exception:
                        return 0.0
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    if denom == 0:
                        return 0.0
                    return float(np.dot(a, b) / denom)

                threshold = self.similarity_threshold
                top_k = self.similarity_top_k

                scored: List[Dict[str, Any]] = []
                for r in rows:
                    emb = r.get("embedding")
                    if not emb:
                        continue
                    sim = cosine(emb, query_emb)
                    candidate_meta = {
                        "retrieval_method": "entities_client_product_embedding",
                        "score": sim,
                    }
                    scored.append(
                        {
                            "p": {"id": r.get("id"), "name": r.get("name")},
                            "_meta": candidate_meta,
                        }
                    )

                scored = [c for c in scored if c["_meta"]["score"] >= threshold]
                scored.sort(
                    key=lambda x: x.get("_meta", {}).get("score", 0), reverse=True
                )
                scored = scored[:top_k]

                logger.info(
                    "Client-side product-embedding fallback returned %d results (threshold=%s top_k=%d)",
                    len(scored),
                    threshold,
                    top_k,
                )
                return scored
            return results
        except Neo4jError as e:
            msg = str(e)
            logger.warning("Server GDS query failed: %s", msg)
            if (
                "gds.similarity" in msg
                or "Unknown function 'gds.similarity.cosine'" in msg
            ):
                # Try alternate GDS function name (some GDS versions expose alpha namespace)
                try:
                    alt_query = query.replace(
                        "gds.similarity.cosine", "gds.alpha.similarity.cosine"
                    )
                    alt_results = self.graph.query(alt_query, params=params)
                    for r in alt_results:
                        if isinstance(r, dict) and "p" in r:
                            r.setdefault("_meta", {})[
                                "retrieval_method"
                            ] = "entities_server_gds_alpha"
                    logger.info(
                        "Alternate server-side GDS function succeeded (gds.alpha.*). Returning %d rows",
                        len(alt_results) if alt_results else 0,
                    )
                    return alt_results
                except Exception as alt_exc:
                    logger.debug("Alternate GDS function attempt failed: %s", alt_exc)
                logger.info(
                    "Falling back to client-side product-embedding similarity computation"
                )
                # If entity-level GDS isn't available, compute a combined embedding for all entities
                # and compare against product embeddings client-side.
                keys = [k for k in query_data.keys() if k != "product"]
                if not keys:
                    return []
                # Combine entity text into one string (e.g. "color: red category: clothes")
                combined_text = " ".join([f"{k}: {query_data[k]}" for k in keys])
                try:
                    # Respect configured model and call embedder
                    query_emb = self.embedder(combined_text)
                except Exception:
                    logger.exception(
                        "Failed to compute embedding for combined entity text"
                    )
                    return []

                # Fetch product embeddings and compute cosine similarity client-side
                rows = self.graph.query(
                    "MATCH (p:Product) WHERE p.embedding IS NOT NULL RETURN p.id AS id, p.name AS name, p.embedding AS embedding"
                )

                def cosine(a, b):
                    try:
                        a = np.array(a, dtype=float)
                        b = np.array(b, dtype=float)
                    except Exception:
                        return 0.0
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    if denom == 0:
                        return 0.0
                    return float(np.dot(a, b) / denom)

                # Use instance-configured threshold/top_k
                threshold = self.similarity_threshold
                top_k = self.similarity_top_k

                scored: List[Dict[str, Any]] = []
                for r in rows:
                    emb = r.get("embedding")
                    if not emb:
                        continue
                    sim = cosine(emb, query_emb)
                    # store score for sorting
                    candidate_meta = {
                        "retrieval_method": "entities_client_product_embedding",
                        "score": sim,
                    }
                    scored.append(
                        {
                            "p": {"id": r.get("id"), "name": r.get("name")},
                            "_meta": candidate_meta,
                        }
                    )

                # Filter and sort
                scored = [c for c in scored if c["_meta"]["score"] >= threshold]
                scored.sort(
                    key=lambda x: x.get("_meta", {}).get("score", 0), reverse=True
                )
                scored = scored[:top_k]

                logger.info(
                    "Client-side product-embedding fallback returned %d results (threshold=%s top_k=%d)",
                    len(scored),
                    threshold,
                    top_k,
                )
                return scored
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
