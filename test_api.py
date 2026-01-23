"""
Simple test script to verify LATS API is working.
Tests with a mock execution engine (no real Java backend needed).
"""

import asyncio
import httpx
from datetime import datetime


async def test_health_check():
    """Test health check endpoint"""
    print("\n" + "=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/v1/lats/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    print("✅ Health check passed")


async def test_root_endpoint():
    """Test root endpoint"""
    print("\n" + "=" * 60)
    print("TEST 2: Root Endpoint")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200

    print("✅ Root endpoint passed")


async def test_sessions_list():
    """Test sessions list endpoint"""
    print("\n" + "=" * 60)
    print("TEST 3: List Sessions")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/v1/lats/sessions")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200

    print("✅ Sessions list passed")


async def test_search_endpoint_validation():
    """Test search endpoint input validation"""
    print("\n" + "=" * 60)
    print("TEST 4: Search Input Validation")
    print("=" * 60)

    # Test with invalid coverage_target (> 1.0)
    invalid_request = {
        "session_id": "test_invalid",
        "function_signature": "int foo(int x)",
        "function_path": "src/test.cpp::foo",
        "function_code": "int foo(int x) { return x; }",
        "coverage_target": 1.5,  # Invalid!
        "max_iterations": 10,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/lats/search", json=invalid_request, timeout=30.0
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 422  # Validation error

    print("✅ Input validation passed")


async def main():
    """Run all tests"""
    print("\n" + "🧪 LATS API Test Suite")
    print("Make sure the server is running: python main.py")
    print()

    try:
        await test_health_check()
        await test_root_endpoint()
        await test_sessions_list()
        await test_search_endpoint_validation()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except httpx.ConnectError:
        print("\n❌ ERROR: Cannot connect to server")
        print("Please start the server first: python main.py")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())
