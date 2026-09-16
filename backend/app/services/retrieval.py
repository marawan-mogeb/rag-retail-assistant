import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.utils.logging_config import logger


class RetrievalService:
    """Loads the persisted Chroma vector store + embedding model once at startup."""

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.embed_model = SentenceTransformer(settings.embedding_model)

        logger.info(f"Loading vector store from: {settings.vector_store_dir}")
        self.client = chromadb.PersistentClient(path=settings.vector_store_dir)
        self.collection = self.client.get_collection(settings.collection_name)
        logger.info(f"Vector store loaded with {self.collection.count()} chunks")

    def retrieve(self, query: str, k: int = 3):
        query_emb = self.embed_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=k)
        return results

    def retrieve_sources(self, query: str, k: int = 3) -> list[str]:
        results = self.retrieve(query, k=k)
        if not results["metadatas"] or not results["metadatas"][0]:
            return []
        return [m["source"] for m in results["metadatas"][0]]
