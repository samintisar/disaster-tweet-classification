"""Contract tests for the disaster tweet classification API."""

import pytest
import requests
import json

BASE_URL = "http://localhost:8000"

class TestHealthEndpoint:
    """Contract tests for /api/health endpoint."""

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

    def test_health_response_has_model_loaded(self):
        """Test that health response has model_loaded field."""
        response = requests.get(f"{BASE_URL}/api/health")
        data = response.json()
        assert 'model_loaded' in data
        assert isinstance(data['model_loaded'], bool)

    def test_health_response_has_service_field(self):
        """Test that health response has service field."""
        response = requests.get(f"{BASE_URL}/api/health")
        data = response.json()
        assert 'service' in data
        assert data['service'] == 'classification'