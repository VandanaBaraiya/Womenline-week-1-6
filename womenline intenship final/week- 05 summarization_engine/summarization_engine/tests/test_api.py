from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

def test_summarize_endpoint_ok():
    payload = {"text": "This is a long enough journal entry about my day and projects." * 5}
    r = client.post("/summarize-journal", json=payload)
    assert r.status_code == 200
    assert "summary" in r.json()

def test_summarize_endpoint_bad_request():
    r = client.post("/summarize-journal", json={"text": "short"})
    assert r.status_code == 400
