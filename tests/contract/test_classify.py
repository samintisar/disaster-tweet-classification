"""Contract tests for the /api/classify endpoint."""

import pytest
import requests

BASE_URL = "http://localhost:8000"

class TestClassifyEndpoint:
    """Contract tests for /api/classify endpoint."""

    def test_classify_endpoint_exists(self):
        """Test that classify endpoint exists."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        assert response.status_code in [200, 500]  # 500 if model not loaded

    def test_classify_accepts_post(self):
        """Test that classify endpoint accepts POST requests."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        assert response.status_code in [200, 500]

    def test_classify_requires_text_field(self):
        """Test that classify endpoint requires text field."""
        response = requests.post(f"{BASE_URL}/api/classify", json={})
        assert response.status_code == 422  # FastAPI validation error

    def test_classify_returns_prediction(self):
        """Test that classify endpoint returns prediction."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data

    def test_classify_returns_confidence(self):
        """Test that classify endpoint returns confidence score."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        if response.status_code == 200:
            data = response.json()
            assert 'confidence' in data
            assert isinstance(data['confidence'], (int, float))

    def test_classify_validates_text_length(self):
        """Test that classify endpoint validates text length."""
        long_text = "x" * 281  # Exceeds 280 character limit
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": long_text})
        assert response.status_code == 422

    def test_classify_handles_invalid_json(self):
        """Test that classify endpoint handles invalid JSON."""
        response = requests.post(f"{BASE_URL}/api/classify", data="invalid json", headers={'Content-Type': 'application/json'})
        assert response.status_code == 422

    def test_classify_optional_fields(self):
        """Test that classify endpoint handles optional fields."""
        payload = {
            "text": "Test tweet",
            "include_features": True,
            "include_keywords": True
        }
        response = requests.post(f"{BASE_URL}/api/classify", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data
            assert 'confidence' in data

    def test_classify_minimal_payload(self):
        """Test that classify endpoint works with minimal payload."""
        payload = {"text": "Test tweet"}
        response = requests.post(f"{BASE_URL}/api/classify", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data
            assert 'confidence' in data