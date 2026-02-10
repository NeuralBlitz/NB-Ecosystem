"""
Quick test script for NB Omnibus Server
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server import create_application, ServerConfig
from fastapi.testclient import TestClient


def test_basic_server():
    """Test basic server functionality"""
    print("Testing NB Omnibus Server...")

    # Create config with cache disabled for testing
    config = ServerConfig(host="127.0.0.1", port=8000, cache_enabled=False, debug=True)

    # Create application
    app = create_application(config=config)

    # Create test client
    client = TestClient(app)

    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "NeuralBlitz" in data["name"]
    print(f"   ✓ Root: {data['name']}")

    # Test health endpoint
    print("\n2. Testing health endpoint...")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    print(f"   ✓ Health: {data['status']}")

    # Test readiness endpoint
    print("\n3. Testing readiness endpoint...")
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    print(f"   ✓ Ready: {data['ready']}")

    # Test docs endpoint
    print("\n4. Testing docs endpoint...")
    response = client.get("/docs")
    assert response.status_code == 200
    print("   ✓ Docs accessible")

    # Test OpenAPI schema
    print("\n5. Testing OpenAPI schema...")
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    print(f"   ✓ OpenAPI schema: {len(schema['paths'])} paths defined")

    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)

    return True


if __name__ == "__main__":
    try:
        test_basic_server()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
