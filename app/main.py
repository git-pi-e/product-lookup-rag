from fastapi import FastAPI
import logging

from .api.routes import router

# Configure root logger for debug output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

app = FastAPI(title="Amazon Product Lookup RAG", version="0.1.0")

app.include_router(router)
