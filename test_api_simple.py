"""Simple API test script."""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test health endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Health endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Health data: {json.dumps(data, indent=2)}")
            return True
    except Exception as e:
        print(f"Health endpoint error: {e}")
    return False

def test_classify_endpoint():
    """Test classify endpoint."""
    try:
        response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test tweet"})
        print(f"Classify endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Classification result: {json.dumps(data, indent=2)}")
            return True
    except Exception as e:
        print(f"Classify endpoint error: {e}")
    return False

def test_batch_classify_endpoint():
    """Test batch classify endpoint."""
    try:
        response = requests.post(f"{BASE_URL}/api/batch-classify", json={"tweets": ["Test tweet 1", "Test tweet 2"]})
        print(f"Batch classify endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Batch result count: {len(data['results'])}")
            return True
    except Exception as e:
        print(f"Batch classify endpoint error: {e}")
    return False

def test_stream_endpoints():
    """Test streaming endpoints."""
    try:
        # Start stream
        response = requests.post(f"{BASE_URL}/api/stream/start", json={"keywords": ["test"], "interval": 60})
        print(f"Stream start endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            stream_id = data['stream_id']
            print(f"Stream started: {stream_id}")

            # Stop stream
            response = requests.post(f"{BASE_URL}/api/stream/stop", json={"stream_id": stream_id})
            print(f"Stream stop endpoint: {response.status_code}")
            return True
    except Exception as e:
        print(f"Stream endpoints error: {e}")
    return False

if __name__ == "__main__":
    print("Testing API endpoints...")

    success_count = 0
    total_tests = 4

    if test_health_endpoint():
        success_count += 1

    if test_classify_endpoint():
        success_count += 1

    if test_batch_classify_endpoint():
        success_count += 1

    if test_stream_endpoints():
        success_count += 1

    print(f"\nTest Results: {success_count}/{total_tests} passed")

    if success_count == total_tests:
        print("All API endpoints are working correctly!")
    else:
        print("Some endpoints failed.")