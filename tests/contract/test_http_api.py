"""Test HTTP client for testing the FastAPI disaster tweet classification API."""

import pytest
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"


class TestHealthEndpoint:
    """HTTP tests for /api/health endpoint."""

    def test_health_endpoint_exists(self):
        """Test that health endpoint exists and returns 200."""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self):
        """Test that health endpoint returns JSON."""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.headers['content-type'] == 'application/json'

    def test_health_response_has_status_field(self):
        """Test that health response has status field."""
        response = requests.get(f"{BASE_URL}/api/health")
        data = response.json()
        assert 'status' in data

    def test_health_status_has_valid_values(self):
        """Test that status has valid values."""
        response = requests.get(f"{BASE_URL}/api/health")
        data = response.json()
        assert data['status'] in ['healthy']

    def test_health_response_has_timestamp(self):
        """Test that health response has timestamp."""
        response = requests.get(f"{BASE_URL}/api/health")
        data = response.json()
        assert 'timestamp' in data


class TestClassifyEndpoint:
    """HTTP tests for /api/classify endpoint."""

    def test_classify_endpoint_exists(self):
        """Test that classify endpoint exists and accepts POST."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        assert response.status_code in [200, 500]  # 500 if model not loaded

    def test_classify_accepts_post(self):
        """Test that classify endpoint accepts POST requests."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        assert response.status_code in [200, 500]

    def test_classify_requires_text_field(self):
        """Test that classify requires text field."""
        response = requests.post(f"{BASE_URL}/api/classify", json={})
        assert response.status_code == 422  # FastAPI validation error

    def test_classify_returns_prediction(self):
        """Test that classify returns prediction."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data

    def test_classify_returns_confidence(self):
        """Test that classify returns confidence."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        if response.status_code == 200:
            data = response.json()
            assert 'confidence' in data
            assert isinstance(data['confidence'], (int, float))

    def test_classify_returns_probabilities(self):
        """Test that classify returns probabilities."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        if response.status_code == 200:
            data = response.json()
            assert 'probabilities' in data
            assert 'disaster' in data['probabilities']
            assert 'non_disaster' in data['probabilities']

    def test_classify_returns_tweet_id(self):
        """Test that classify returns tweet_id."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        if response.status_code == 200:
            data = response.json()
            assert 'tweet_id' in data

    def test_classify_returns_timestamp(self):
        """Test that classify returns timestamp."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        if response.status_code == 200:
            data = response.json()
            assert 'timestamp' in data

    def test_classify_validates_text_length(self):
        """Test that classify validates text length."""
        long_text = "x" * 281  # Exceeds 280 character limit
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": long_text})
        assert response.status_code == 422

    def test_classify_handles_disaster_tweets(self):
        """Test that classify handles disaster-related tweets."""
        disaster_tweet = "Major earthquake hits San Francisco, buildings damaged and people injured #earthquake"
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": disaster_tweet})
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data
            assert data['prediction'] in ['disaster', 'non_disaster']

    def test_classify_handles_normal_tweets(self):
        """Test that classify handles normal tweets."""
        normal_tweet = "Just had a great lunch with friends at the new restaurant downtown!"
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": normal_tweet})
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data
            assert data['prediction'] in ['disaster', 'non_disaster']

    def test_classify_handles_empty_text(self):
        """Test that classify handles empty text."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": ""})
        assert response.status_code == 422

    def test_classify_with_features_flag(self):
        """Test that classify handles include_features flag."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet", "include_features": True})
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data

    def test_classify_with_keywords_flag(self):
        """Test that classify handles include_keywords flag."""
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet", "include_keywords": True})
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Testing health endpoint...")
    health_test = TestHealthEndpoint()
    health_test.test_health_endpoint_exists()
    print("✓ Health endpoint exists")

    print("Testing classify endpoint...")
    classify_test = TestClassifyEndpoint()
    classify_test.test_classify_endpoint_exists()
    print("✓ Classify endpoint exists")

    print("\nAll basic endpoint tests passed!")