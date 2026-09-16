import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas.query import DetectionInfo, QueryResponse
from app.utils.logging_config import logger

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    question: str = Form(...),
    image: UploadFile | str | None = File(default=None),
):
    if not question or not question.strip():
        raise HTTPException(status_code=422, detail="`question` must not be empty")

    # Some clients (e.g. Swagger UI's "send empty value") submit an empty
    # string instead of omitting the field entirely — treat that as "no image".
    if isinstance(image, str) or image is None or not getattr(image, "filename", None):
        image = None

    retrieval_service = request.app.state.retrieval_service
    generation_service = request.app.state.generation_service
    vision_service = request.app.state.vision_service

    detection_result = None
    detection_info = None

    # If an image was uploaded, run YOLO detection first so it can be
    # injected into the RAG prompt as extra context.
    if image is not None:
        suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            tmp_path = tmp.name
        try:
            detection_result = vision_service.detect(tmp_path)
            detection_info = DetectionInfo(**detection_result)
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            raise HTTPException(status_code=500, detail="Image detection failed") from e
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Retrieve relevant chunks
    retrieved = retrieval_service.retrieve(question, k=3)
    sources = retrieval_service.retrieve_sources(question, k=3)

    # Build prompt (with detection context if present) and call the LLM
    prompt = generation_service.build_prompt(question, retrieved, detection_context=detection_result)
    try:
        answer = generation_service.generate(prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return QueryResponse(answer=answer, sources=sources, detection=detection_info)