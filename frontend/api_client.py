"""
Thin wrapper around the FastAPI backend's /query and /health endpoints.
Keeps all HTTP details (URLs, multipart encoding, timeouts) out of app.py.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 60  # seconds — generous since local LLM inference can be slow


class APIError(Exception):
    """Raised when the backend returns an error or is unreachable."""
    pass


def check_health() -> bool:
    """Returns True if the backend is reachable and healthy."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def send_query(question: str, image_bytes: bytes | None = None, image_filename: str = "upload.jpg") -> dict:
    """
    Calls POST /query with a question and an optional image.

    Returns the parsed JSON response: {"answer": ..., "sources": [...], "detection": {...} | None}
    Raises APIError with a user-friendly message on failure.
    """
    data = {"question": question}
    files = None
    if image_bytes is not None:
        files = {"image": (image_filename, image_bytes, "image/jpeg")}

    try:
        resp = requests.post(
            f"{API_BASE_URL}/query",
            data=data,
            files=files,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as e:
        raise APIError(
            f"Could not connect to the backend at {API_BASE_URL}. "
            "Is the FastAPI server running?"
        ) from e
    except requests.exceptions.Timeout as e:
        raise APIError(
            "The backend took too long to respond. The local LLM may be slow — try again."
        ) from e

    if resp.status_code == 422:
        raise APIError("Your question could not be processed. Please enter a valid question.")
    if resp.status_code == 502:
        detail = resp.json().get("detail", "The LLM service is unavailable.")
        raise APIError(f"LLM error: {detail}")
    if resp.status_code != 200:
        raise APIError(f"Unexpected error from backend (status {resp.status_code}): {resp.text}")

    return resp.json()
