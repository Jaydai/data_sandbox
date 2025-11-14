import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    """Test endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"


def test_health_check():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_classify_work_message():
    """Test classification message professionnel"""
    payload = {
        "user_id": "user_123",
        "organization_id": "org_456",
        "content": "Peux-tu analyser les données de ventes du Q4 pour notre présentation au board ?",
        "detect_pii": True,
        "assess_quality": True,
        "analyze_risks": True
    }
    
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["work"]["is_work"] == True
    assert data["topic"]["topic"] in ["TECHNICAL_HELP", "PRACTICAL_GUIDANCE"]
    assert data["processing_time_ms"] > 0


def test_classify_non_work_message():
    """Test classification message personnel"""
    payload = {
        "user_id": "user_123",
        "organization_id": "org_456",
        "content": "Donne-moi une recette de crêpes facile.",
        "detect_pii": False,
        "assess_quality": False,
        "analyze_risks": False
    }
    
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["work"]["is_work"] == False
    assert data["pii_detection"] is None  # Désactivé


def test_classify_with_pii():
    """Test détection PII"""
    payload = {
        "user_id": "user_123",
        "organization_id": "org_456",
        "content": "Mon email est john.doe@example.com et mon téléphone est 06 12 34 56 78",
        "detect_pii": True,
        "assess_quality": False,
        "analyze_risks": False
    }
    
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["pii_detection"]["has_pii"] == True
    assert len(data["pii_detection"]["pii_types"]) > 0


def test_bulk_classification():
    """Test classification en batch"""
    payload = {
        "organization_id": "org_456",
        "async_processing": False,
        "messages": [
            {
                "user_id": "user_1",
                "organization_id": "org_456",
                "content": "Rédige un email pro"
            },
            {
                "user_id": "user_2",
                "organization_id": "org_456",
                "content": "Recette de pâtes"
            }
        ]
    }
    
    response = client.post("/classify/bulk", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_invalid_request():
    """Test requête invalide"""
    payload = {
        "user_id": "user_123"
        # Manque organization_id et content
    }
    
    response = client.post("/classify", json=payload)
    assert response.status_code == 422  # Validation error