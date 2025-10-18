from fastapi import FastAPI

from .api.routes import router


app = FastAPI(title="Amazon Product Lookup RAG", version="0.1.0")

app.include_router(router)
