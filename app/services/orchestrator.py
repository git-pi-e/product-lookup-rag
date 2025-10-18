import json
from typing import Any, Dict, List

from .llm import LLMService
from .retrieval import RetrievalService


class Orchestrator:
    def __init__(
        self, llm: LLMService | None = None, retrieval: RetrievalService | None = None
    ):
        self.llm = llm or LLMService()
        self.retrieval = retrieval or RetrievalService(embedder=self.llm.embed_text)

    def run(self, prompt: str, similar_items_limit: int = 10) -> Dict[str, Any]:
        extracted_json = self.llm.extract_entities(prompt)
        results = self.retrieval.query_by_entities(extracted_json)

        matches: List[Dict[str, Any]] = []
        for r in results:
            matches.append({"id": r["p"]["id"], "name": r["p"]["name"]})

        if not matches:
            fallback = self.retrieval.query_by_prompt_similarity(prompt)
            for r in fallback:
                matches.append({"id": r["p"]["id"], "name": r["p"]["name"]})

        # Compute similar items and cap at 5 by default
        similar: List[Dict[str, Any]] = []
        for m in matches:
            similar.extend(self.retrieval.similar_items(m["id"]))

        return {
            "query": prompt,
            "extracted_entities": json.loads(extracted_json or "{}"),
            "matches": matches,
            "similar_items": similar[: min(5, similar_items_limit or 5)],
        }
