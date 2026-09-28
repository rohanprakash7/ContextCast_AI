"""
Pytest Configuration and Shared Fixtures
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_video_id():
    """Fixture providing test video ID"""
    return "kY14KfZQ1TI"


@pytest.fixture
def api_client():
    """Fixture providing FastAPI test client"""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_question():
    """Fixture providing sample question"""
    return "What is deep learning?"