import ollama

from app.core.config import settings
from app.utils.logging_config import logger


class GenerationService:
    """Wraps prompt construction + the Ollama LLM call."""

    def __init__(self):
        self.model = settings.ollama_model
        self.client = ollama.Client(host=settings.ollama_host)
        logger.info(f"GenerationService ready, using Ollama model: {self.model}")

    @staticmethod
    def build_prompt(question: str, retrieved: dict, detection_context: dict | None = None) -> str:
        context_blocks = []
        docs = retrieved.get("documents", [[]])[0]
        metas = retrieved.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            context_blocks.append(f"[Source: {meta['source']}]\n{doc}")
        context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        detection_line = (
            f"\nDetected shelf info: {detection_context}\n" if detection_context else ""
        )

        return f"""Answer the question using ONLY the context below. Cite the source filename for each claim.
If the context does not contain the answer, say you don't have enough information rather than guessing.
{detection_line}
Context:
{context}

Question: {question}

Answer (with citations):"""

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_host} with model "
                f"'{self.model}'. Is `ollama serve` running and is the model pulled "
                f"(`ollama pull {self.model}`)?"
            ) from e
