"""
Tests mock out RetrievalService / GenerationService / VisionService so that
running pytest doesn't require a real Ollama server, YOLO weights, or vector
store to be present. This keeps tests fast and independent of local setup.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.main.RetrievalService") as MockRetrieval, \
         patch("app.main.GenerationService") as MockGeneration, \
         patch("app.main.VisionService") as MockVision:

        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = {
            "documents": [["Sample policy text."]],
            "metadatas": [[{"source": "sample.txt"}]],
        }
        mock_retrieval.retrieve_sources.return_value = ["sample.txt"]
        MockRetrieval.return_value = mock_retrieval

        mock_generation = MagicMock()
        mock_generation.build_prompt.return_value = "prompt"
        mock_generation.generate.return_value = "This is a grounded answer. [Source: sample.txt]"
        MockGeneration.return_value = mock_generation

        MockVision.return_value = MagicMock()

        from app.main import app  # imported after patches so lifespan uses mocks

        with TestClient(app) as test_client:
            yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_happy_path(client):
    response = client.post("/query", data={"question": "What is the return policy?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert body["sources"] == ["sample.txt"]
    assert body["detection"] is None


def test_query_invalid_empty_question(client):
    response = client.post("/query", data={"question": ""})
    assert response.status_code == 422
