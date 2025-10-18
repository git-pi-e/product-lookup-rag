from pydantic import BaseModel


class QueryRequest(BaseModel):
    prompt: str
    similar_items_limit: int | None = 10


class HealthResponse(BaseModel):
    status: str
