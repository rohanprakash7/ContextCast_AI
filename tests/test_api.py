"""
Test API Endpoints
Integration tests for FastAPI backend
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test GET /health"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    
    print(f"\n✓ Health endpoint working: {data}")


def test_root_endpoint():
    """Test GET / (root)"""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
    
    print(f"\n✓ Root endpoint working")


def test_video_status_endpoint():
    """Test GET /index/status/{video_id}"""
    video_id = "kY14KfZQ1TI"
    response = client.get(f"/index/status/{video_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == video_id
    assert "is_indexed" in data
    
    print(f"\n✓ Status endpoint working: indexed={data['is_indexed']}")


def test_index_endpoint_invalid_video_id():
    """Test POST /index with invalid video ID"""
    response = client.post(
        "/index",
        json={"video_id": "invalid"}  # Too short
    )
    
    assert response.status_code == 422  # Validation error
    print("\n✓ Validation working for invalid video ID")


def test_query_endpoint_invalid_question():
    """Test POST /query with invalid question"""
    response = client.post(
        "/query",
        json={
            "question": "Hi",  # Too short (< 3 chars)
            "video_id": "kY14KfZQ1TI"
        }
    )
    
    assert response.status_code == 422  # Validation error
    print("\n✓ Validation working for short question")


def test_query_endpoint_invalid_retriever():
    """Test POST /query with invalid retriever type"""
    response = client.post(
        "/query",
        json={
            "question": "What is this?",
            "video_id": "kY14KfZQ1TI",
            "retriever_type": "invalid_type"
        }
    )
    
    assert response.status_code == 422  # Validation error
    print("\n✓ Validation working for invalid retriever type")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])