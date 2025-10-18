from fastapi import APIRouter

from .schemas import QueryRequest, HealthResponse
from ..services.orchestrator import Orchestrator
from ..services.agent import agent_run
from ..services.agent import agent_run_structured


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
