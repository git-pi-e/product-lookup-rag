import json
from typing import Any, Dict, Optional

from openai import OpenAI

from ..config import get_settings
from ..core.entities import entity_types, relation_types


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


class LLMService:
    def __init__(self, client: Optional[OpenAI] = None, model: str = "gpt-4o") -> None:
        settings = get_settings()
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = model
        self.system_prompt = build_system_prompt()
        self.embeddings_model = settings.embeddings_model

    def extract_entities(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content or "{}"

    def embed_text(self, text: str):
        result = self.client.embeddings.create(model=self.embeddings_model, input=text)
        return result.data[0].embedding
