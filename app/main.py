from fastapi import FastAPI, Request
import logging
import uuid
import contextvars

from .api.routes import router

# Context var to hold request id per async context
request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="N/A"
)


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx_var.get("N/A")
        return True


# Configure root logger for sane defaults (INFO). Include request id in format.
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(request_id)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Attach RequestIDFilter to existing handlers so the extra field is available
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIDFilter())

# Reduce verbosity from noisy third-party libraries
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.io").setLevel(logging.WARNING)
logging.getLogger("neo4j.pool").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

app = FastAPI(title="Amazon Product Lookup RAG", version="0.1.0")


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    rid = str(uuid.uuid4())
    token = request_id_ctx_var.set(rid)
    try:
        response = await call_next(request)
        # Echo the request id back to the client for tracing
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        request_id_ctx_var.reset(token)


app.include_router(router)
