import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import get_settings
from .graph import get_graph
from .core.entities import (
    entity_types,
    relation_types,
    entity_relationship_match,
)


def build_system_prompt() -> str:
    return f"""
    You are a helpful agent designed to fetch information from a graph database.

    The graph database links products to the following entity types:
    {json.dumps(entity_types)}

    Each link has one of the following relationships:
    {json.dumps(relation_types)}

    Depending on the user prompt, determine if it is possible to answer with the graph database.

    The graph database can match products with multiple relationships to several entities.

    Example user input:
    "Which blue clothing items are suitable for adults?"

    There are three relationships to analyse:
    1. The mention of the blue color means we will search for a color similar to "blue"
    2. The mention of the clothing items means we will search for a category similar to "clothing"
    3. The mention of adults means we will search for an age_group similar to "adults"

    Return a json object following the following rules:
    For each relationship to analyse, add a key value pair with the key being an exact match for one of the entity types provided, and the value being the value relevant to the user query.

    For the example provided, the expected output would be:
    {{
        "color": "blue",
        "category": "clothing",
        "age_group": "adults"
    }}

    If there are no relevant entities in the user prompt, return an empty json object.
    """


class RAGService:
    def __init__(self, client: Optional[OpenAI] = None):
        self.settings = get_settings()
        self.client = client or OpenAI(api_key=self.settings.openai_api_key)
        self.graph = get_graph()
        self.embeddings_model = self.settings.embeddings_model
        self.system_prompt = build_system_prompt()

    # 1) Extract entities from the query via JSON-structured LLM output
    def define_query(self, prompt: str, model: str = "gpt-4o") -> str:
        completion = self.client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content or "{}"

    # 2) Low-level embedding call
    def create_embedding(self, text: str) -> List[float]:
        result = self.client.embeddings.create(model=self.embeddings_model, input=text)
        return result.data[0].embedding  # type: ignore[no-any-return]

    # 3) Build Cypher query with cosine similarity filters against entity embeddings
    def create_query(self, extracted_json: str, threshold: float = 0.81) -> str:
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

    # 4) Execute entity-based retrieval
    def query_graph(self, extracted_json: str) -> List[Dict[str, Any]]:
        embeddings_params: Dict[str, Any] = {}
        query = self.create_query(extracted_json)
        query_data = json.loads(extracted_json)
        for key, val in query_data.items():
            if key != "product":
                embeddings_params[f"{key}Embedding"] = self.create_embedding(str(val))
        return self.graph.query(query, params=embeddings_params)

    # 5) Fallback: product-level similarity search
    def similarity_search(
        self, prompt: str, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        embedding = self.create_embedding(prompt)
        query = """
            WITH $embedding AS inputEmbedding
            MATCH (p:Product)
            WHERE gds.similarity.cosine(inputEmbedding, p.embedding) > $threshold
            RETURN p
            """
        return self.graph.query(
            query, params={"embedding": embedding, "threshold": threshold}
        )

    # 6) Similar items by shared category and common entities
    def query_similar_items(
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

    # 7) High-level API orchestrator
    def run(self, prompt: str, similar_items_limit: int = 10) -> Dict[str, Any]:
        extracted = self.define_query(prompt)
        results = self.query_graph(extracted)

        matches: List[Dict[str, Any]] = []
        for r in results:
            matches.append({"id": r["p"]["id"], "name": r["p"]["name"]})

        if not matches:
            fallback = self.similarity_search(prompt)
            for r in fallback:
                matches.append({"id": r["p"]["id"], "name": r["p"]["name"]})

        similar: List[Dict[str, Any]] = []
        for m in matches:
            similar.extend(self.query_similar_items(m["id"]))

        return {
            "query": prompt,
            "extracted_entities": json.loads(extracted or "{}"),
            "matches": matches,
            "similar_items": (
                similar[:similar_items_limit] if similar_items_limit else similar
            ),
        }
