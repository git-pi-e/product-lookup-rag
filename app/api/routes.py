from fastapi import APIRouter, Request
from pydantic import BaseModel

from .schemas import QueryRequest, HealthResponse
from ..services.orchestrator import Orchestrator
from ..services.agent import agent_run
from ..services.agent import agent_run_structured
from ..graph import get_graph


class CypherRequest(BaseModel):
    cypher: str
    params: dict | None = None


router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/query")
def query(req: QueryRequest):
    orchestrator = Orchestrator()
    return orchestrator.run(
        req.prompt, similar_items_limit=req.similar_items_limit or 10
    )


@router.post("/agent")
def agent_endpoint(req: QueryRequest):
    output = agent_run(req.prompt)
    return {"output": output}


@router.post("/agent/structured")
def agent_structured(req: QueryRequest):
    return agent_run_structured(req.prompt)


@router.post("/debug/cypher")
def debug_cypher(req: CypherRequest, request: Request):
    """Execute a read-only Cypher query and return rows. Use only in trusted/dev environments."""
    graph = get_graph()
    params = req.params or {}
    rows = graph.query(req.cypher, params=params)
    return {"x_request_id": request.headers.get("x-request-id"), "rows": rows}
