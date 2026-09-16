from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService
from app.services.vision import VisionService
from app.utils.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load vector store, embedding model, YOLO model, and Ollama client ONCE.
    logger.info("Starting up: loading models and vector store...")
    app.state.retrieval_service = RetrievalService()
    app.state.generation_service = GenerationService()
    app.state.vision_service = VisionService()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="RAG Retail Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
