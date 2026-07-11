"""
Unit and integration tests for FastAPI backend routes.
Tests cover:
1. Health check endpoints (GET / and GET /health)
2. File validation in POST /analyze (type, size, empty)
3. Synchronous pipeline execution with mock model and mocked LLM calls
4. GET /history and GET /results/{run_id} endpoints
"""

import os
import io
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

@pytest.fixture(scope="module", autouse=True)
def set_env():
    # Save original state if exists
    old_val = os.environ.get("NEUROTRIAGE_DEV_MODE")
    os.environ["NEUROTRIAGE_DEV_MODE"] = "true"
    yield
    # Restore original state
    if old_val is not None:
        os.environ["NEUROTRIAGE_DEV_MODE"] = old_val
    else:
        del os.environ["NEUROTRIAGE_DEV_MODE"]


from app.main import app
from app.db.models import init_db, get_db_session, AnalysisRun, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup test client
client = TestClient(app)

# Use a separate SQLite database file for testing to avoid modifying development DB
TEST_DATABASE_URL = "sqlite:///test_neurotriage.db"


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Initialize a clean database for testing."""
    import app.db.models as db_models
    
    # Override engine and session local factory in db module
    test_engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=test_engine)
    
    db_models.engine = test_engine
    db_models.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    # Initialize DB (creates schemas)
    db_models.init_db = MagicMock()  # Mock out startup init_db to prevent overwriting
    
    yield
    
    # Cleanup tables
    Base.metadata.drop_all(bind=test_engine)
    
    # Remove test DB file
    if os.path.exists("test_neurotriage.db"):
        try:
            os.remove("test_neurotriage.db")
        except Exception:
            pass


@pytest.fixture
def clean_db():
    """Clean all records from tables before each test."""
    session = get_db_session()
    try:
        session.query(AnalysisRun).delete()
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_root_endpoint():
    """Verify GET / health check endpoint returns 200 and OK status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "NeuroTriage" in data["message"]


def test_health_check_endpoint():
    """Verify GET /health endpoint returns 200 and health info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database" in data
    assert "environment" in data


def test_analyze_invalid_mime_type():
    """Verify POST /analyze rejects non-JPEG/PNG file formats with HTTP 422."""
    files = {"file": ("test.txt", b"some text content", "text/plain")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 422
    assert "Invalid file format" in response.json()["detail"]


def test_analyze_empty_file():
    """Verify POST /analyze rejects empty files with HTTP 422."""
    files = {"file": ("test.png", b"", "image/png")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_analyze_file_size_limit():
    """Verify POST /analyze rejects files exceeding 20MB with HTTP 422."""
    # Mocking larger than 20MB file
    large_bytes = b"0" * (20 * 1024 * 1024 + 1)
    files = {"file": ("test.png", large_bytes, "image/png")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 422
    assert "exceeds maximum of 20MB" in response.json()["detail"]


@patch("app.agents.drafting.call_llm_with_fallback")
@patch("app.agents.orchestrator.call_llm_with_fallback")
def test_analyze_successful_flow(mock_orch_llm, mock_draft_llm, clean_db):
    """
    Verify POST /analyze with a valid image executes successfully.
    
    Checks:
    - Pipeline completes and returns HTTP 200
    - All fields of AnalysisResult are present and correct
    - DB record is updated to status='completed'
    """
    # Configure mock LLM returns
    mock_draft_llm.return_value = (
        "# Classification\nPredicted Class: glioma\n\n# Confidence\nConfidence level is high.\n\n"
        "# Visual Evidence\nGrad-CAM heatmaps show temporal activation.\n\n# Similar Cases\nNo cases found.\n\n"
        "# Clinical Notes\nNotes about glioma.\n\n# Caveats\nResearch prototype disclaimer.",
        "groq/llama-3.1-8b-instant"
    )
    
    mock_orch_llm.return_value = (
        "Case routed to urgent because of high-confidence glioma.",
        "groq/llama-3.1-8b-instant"
    )
    
    # Load real test image bytes
    test_image_path = "data/Testing/glioma/Te-gl_1.jpg"
    with open(test_image_path, "rb") as f:
        test_png_bytes = f.read()
    
    files = {"file": ("scan.jpg", test_png_bytes, "image/jpeg")}
    response = client.post("/analyze", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "completed"
    assert "run_id" in data
    assert data["classification"] in ["glioma", "meningioma", "notumor", "pituitary"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["predictions"]) == 4
    assert len(data["gradcam_image"]) > 0
    assert "Classification" in data["draft_report"]
    assert "Caveats" in data["draft_report"]
    assert data["routing"] in ["urgent", "auto-clear", "needs-review"]
    assert len(data["justification"]) > 0
    assert len(data["reasoning_trace"]) == 4
    assert data["critic_revision_count"] >= 0
    assert "vision" in data["models_used"]
    assert "drafting" in data["models_used"]
    assert "orchestrator" in data["models_used"]
    
    # Verify DB entry
    session = get_db_session()
    db_record = session.query(AnalysisRun).filter_by(id=data["run_id"]).first()
    assert db_record is not None
    assert db_record.status == "completed"
    assert db_record.top_class == data["classification"]
    assert db_record.routing == data["routing"]
    session.close()


@patch("app.agents.drafting.call_llm_with_fallback")
@patch("app.agents.orchestrator.call_llm_with_fallback")
def test_history_endpoint(mock_orch_llm, mock_draft_llm, clean_db):
    """Verify GET /history returns past analyses in reverse chronological order."""
    # Seed a record manually
    session = get_db_session()
    run1 = AnalysisRun(
        id="run-1",
        status="completed",
        original_filename="scan1.png",
        top_class="glioma",
        top_confidence=0.85,
        predictions_json=json.dumps({"glioma": 0.85, "meningioma": 0.1, "notumor": 0.03, "pituitary": 0.02}),
        routing="urgent",
        critic_revision_count=0
    )
    run2 = AnalysisRun(
        id="run-2",
        status="completed",
        original_filename="scan2.png",
        top_class="notumor",
        top_confidence=0.95,
        predictions_json=json.dumps({"glioma": 0.01, "meningioma": 0.01, "notumor": 0.95, "pituitary": 0.03}),
        routing="auto-clear",
        critic_revision_count=0
    )
    session.add(run1)
    session.add(run2)
    session.commit()
    session.close()
    
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data) == 2
    # Check sorting (by created_at desc)
    # run2 was added second, so it should be first in results
    assert data[0]["run_id"] == "run-2"
    assert data[0]["top_class"] == "notumor"
    assert data[0]["routing"] == "auto-clear"
    
    assert data[1]["run_id"] == "run-1"
    assert data[1]["top_class"] == "glioma"
    assert data[1]["routing"] == "urgent"


def test_results_by_id_endpoint_success(clean_db):
    """Verify GET /results/{run_id} returns 200 and full result if completed."""
    session = get_db_session()
    run = AnalysisRun(
        id="test-run-id",
        status="completed",
        original_filename="scan.png",
        top_class="meningioma",
        top_confidence=0.90,
        predictions_json=json.dumps({"glioma": 0.05, "meningioma": 0.90, "notumor": 0.03, "pituitary": 0.02}),
        gradcam_image_b64="gradcam_base64_data",
        draft_report="Full report text",
        routing="needs-review",
        justification="Needs review because it is meningioma.",
        reasoning_trace_json=json.dumps([{"agent": "vision", "summary": "Detected meningioma", "key_evidence": "90%"}])
    )
    session.add(run)
    session.commit()
    session.close()
    
    response = client.get("/results/test-run-id")
    assert response.status_code == 200
    data = response.json()
    
    assert data["run_id"] == "test-run-id"
    assert data["status"] == "completed"
    assert data["classification"] == "meningioma"
    assert data["confidence"] == 0.90
    assert data["gradcam_image"] == "gradcam_base64_data"
    assert data["draft_report"] == "Full report text"
    assert data["routing"] == "needs-review"
    assert data["justification"] == "Needs review because it is meningioma."
    assert len(data["reasoning_trace"]) == 1


def test_results_by_id_not_found():
    """Verify GET /results/{run_id} returns HTTP 404 if the run ID does not exist."""
    response = client.get("/results/non-existent-id")
    assert response.status_code == 404
    assert "Analysis run not found" in response.json()["detail"]


def test_results_by_id_incomplete(clean_db):
    """Verify GET /results/{run_id} returns HTTP 404 if the run is pending/failed."""
    session = get_db_session()
    run = AnalysisRun(
        id="pending-run-id",
        status="pending"
    )
    session.add(run)
    session.commit()
    session.close()
    
    response = client.get("/results/pending-run-id")
    assert response.status_code == 404
    assert "not yet completed" in response.json()["detail"]
