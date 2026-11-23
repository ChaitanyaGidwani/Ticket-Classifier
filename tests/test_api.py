"""Integration tests for FastAPI REST endpoints."""
from fastapi.testclient import TestClient
import pytest

from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_root(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "service" in data
    assert "huggingface_model" in data
    assert "labels" in data


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "thresholds" in data
    assert "total_parameters" in data


def test_api_classify_valid(client):
    payload = {"text": "I was charged twice on my invoice.", "check_duplicates": False}
    res = client.post("/classify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["primary"] == "Billing"
    assert "Billing" in data["labels"]
    assert "confidence" in data
    assert "calibrated_confidence" in data


def test_api_classify_empty_error(client):
    # Empty string should return 422 or 400
    res = client.post("/classify", json={"text": "   "})
    assert res.status_code in [400, 422]


def test_api_batch_classify(client):
    payload = {
        "texts": [
            "The app crashes every time I launch it",
            "Please cancel my subscription and close my account",
        ],
        "check_duplicates": False,
        "detect_batch_duplicates": True,
    }
    res = client.post("/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["primary"] == "Technical Support"
    assert data["results"][1]["primary"] == "Cancellation"


def test_api_duplicates_search(client):
    payload = {
        "texts": [
            "I was charged twice for my subscription this month",
            "My card got billed two times for my subscription this month",
            "Would love a dark mode theme in the app",
        ],
        "threshold": 0.65,
    }
    res = client.post("/duplicates", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_candidates"] == 3
    assert len(data["duplicate_pairs"]) >= 1
    pair = data["duplicate_pairs"][0]
    assert pair["similarity"] >= 0.65


def test_api_thresholds_workflow(client):
    # GET
    res_get = client.get("/thresholds")
    assert res_get.status_code == 200
    cfg = res_get.json()
    assert "label_threshold" in cfg

    # PUT
    cfg["label_threshold"] = 0.33
    res_put = client.put("/thresholds", json=cfg)
    assert res_put.status_code == 200
    assert res_put.json()["label_threshold"] == 0.33

    # POST alias
    cfg["label_threshold"] = 0.31
    res_post = client.post("/thresholds", json=cfg)
    assert res_post.status_code == 200
    assert res_post.json()["label_threshold"] == 0.31

    # Restore
    cfg["label_threshold"] = 0.30
    client.put("/thresholds", json=cfg)


def test_api_reset_history(client):
    res = client.post("/history/reset")
    assert res.status_code == 200
    assert res.json()["stored_tickets_count"] == 0


def test_api_batch_empty_item_error(client):
    # Empty string inside batch should return 400
    payload = {"texts": ["Valid ticket", "   ", "Another ticket"]}
    res = client.post("/batch", json=payload)
    assert res.status_code == 400
    assert "index 1" in res.json()["detail"]


def test_api_duplicates_empty_item_error(client):
    payload = {"texts": ["Valid ticket", "   "]}
    res = client.post("/duplicates", json=payload)
    assert res.status_code == 400
    assert "index 1" in res.json()["detail"]


def test_api_evaluate(client):
    res = client.get("/evaluate")
    assert res.status_code == 200
    data = res.json()
    assert "exact_match" in data
    assert "primary_acc" in data
    assert "macro_f1" in data
    assert "overall" in data
