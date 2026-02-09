#!/usr/bin/env python3
"""
NeuralBlitz Omnibus Router - Test Script
Verifies API endpoints and functionality
"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"
API_KEY = "nb_pat_xxxxxxxxxxxxxxxxxxxx"


async def test_endpoints():
    """Test all API endpoints."""
    print("🧪 NeuralBlitz API Test Suite")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        headers = {"X-API-Key": API_KEY}

        # Health check
        print("\n1. Testing health check...")
        try:
            r = await client.get(f"{BASE_URL}/health")
            print(f"   Status: {r.status_code}")
            print(f"   Response: {r.json()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Capabilities
        print("\n2. Testing capabilities...")
        try:
            r = await client.get(f"{BASE_URL}/api/v1/capabilities", headers=headers)
            print(f"   Status: {r.status_code}")
            print(f"   Response: {r.json()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Core process
        print("\n3. Testing core process...")
        try:
            r = await client.post(
                f"{BASE_URL}/api/v1/core/process",
                headers=headers,
                json={"input_data": [0.1, 0.2, 0.3]},
            )
            print(f"   Status: {r.status_code}")
            print(f"   Response: {r.json()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Quantum simulate
        print("\n4. Testing quantum simulate...")
        try:
            r = await client.post(
                f"{BASE_URL}/api/v1/quantum/simulate",
                headers=headers,
                json={"qubits": 4, "operations": 100, "coherence": 0.9},
            )
            print(f"   Status: {r.status_code}")
            print(f"   Response: {r.json()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Consciousness level
        print("\n5. Testing consciousness level...")
        try:
            r = await client.get(
                f"{BASE_URL}/api/v1/consciousness/level", headers=headers
            )
            print(f"   Status: {r.status_code}")
            print(f"   Response: {r.json()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Detailed health
        print("\n6. Testing detailed health...")
        try:
            r = await client.get(f"{BASE_URL}/health/detailed")
            print(f"   Status: {r.status_code}")
            print(f"   Response: {r.json()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Prometheus metrics
        print("\n7. Testing Prometheus metrics...")
        try:
            r = await client.get(f"{BASE_URL}/metrics")
            print(f"   Status: {r.status_code}")
            print(f"   Response (first 500 chars): {r.text[:500]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # WebSocket connections
        print("\n8. Testing WebSocket connections endpoint...")
        try:
            r = await client.get(f"{BASE_URL}/api/v1/ws/connections")
            print(f"   Status: {r.status_code}")
            print(f"   Response: {r.json()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 50)
    print("✅ Test suite completed!")


if __name__ == "__main__":
    asyncio.run(test_endpoints())
