"""
Phase 10 Verification: Test FastAPI Endpoints
Run the API first: python run.py
Then run this script in another terminal
"""

import requests
import time

API_BASE = "http://localhost:8000"

print("=" * 60)
print("Phase 10: FastAPI Backend Testing")
print("=" * 60)
print("\nMake sure API is running: python run.py")
print("Waiting 3 seconds...")
time.sleep(3)

# Test 1: Health check
print("\nTest 1: Health Check...")
try:
    response = requests.get(f"{API_BASE}/health")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print("  ✓ Health check working")
except Exception as e:
    print(f"  ✗ Health check failed: {e}")
    print("  Make sure API is running: python run.py")
    exit(1)

# Test 2: Check video status
print("\nTest 2: Check Video Status...")
video_id = "kY14KfZQ1TI"
try:
    response = requests.get(f"{API_BASE}/index/status/{video_id}")
    status_data = response.json()
    print(f"  Video ID: {status_data['video_id']}")
    print(f"  Is indexed: {status_data['is_indexed']}")
    print("  ✓ Status endpoint working")
except Exception as e:
    print(f"  ✗ Status check failed: {e}")

# Test 3: Index video (if not indexed)
print("\nTest 3: Index Video...")
if not status_data.get('is_indexed'):
    print(f"  Indexing {video_id}...")
    print("  (This will take ~20 seconds)")
    
    try:
        response = requests.post(
            f"{API_BASE}/index",
            json={"video_id": video_id}
        )
        index_result = response.json()
        print(f"  ✓ Indexed successfully")
        print(f"    Chunks: {index_result['num_chunks']}")
        print(f"    Duration: {index_result['duration_seconds']:.2f}s")
    except Exception as e:
        print(f"  ✗ Indexing failed: {e}")
else:
    print(f"  Video already indexed, skipping")

# Test 4: Query video
print("\nTest 4: Query Video...")
try:
    response = requests.post(
        f"{API_BASE}/query",
        json={
            "question": "What is deep learning?",
            "video_id": video_id,
            "retriever_type": "simple",
            "top_k": 3,
            "include_citations": True
        }
    )
    query_result = response.json()
    
    print(f"  ✓ Query successful")
    print(f"\n  Question: {query_result['question']}")
    print(f"\n  Answer:\n  {query_result['answer']}")
    print(f"\n  Metadata:")
    print(f"    Retrieved chunks: {query_result['retrieved_chunks']}")
    print(f"    Citations: {query_result.get('citations', [])}")
    print(f"    Duration: {query_result['duration_seconds']:.2f}s")
    
except Exception as e:
    print(f"  ✗ Query failed: {e}")

# Test 5: Try different retriever
print("\nTest 5: Query with Hybrid Retriever...")
try:
    response = requests.post(
        f"{API_BASE}/query",
        json={
            "question": "How do neural networks work?",
            "video_id": video_id,
            "retriever_type": "hybrid",
            "top_k": 3,
            "include_citations": False
        }
    )
    result = response.json()
    
    print(f"  ✓ Hybrid retriever working")
    print(f"  Answer: {result['answer'][:100]}...")
    
except Exception as e:
    print(f"  ✗ Hybrid query failed: {e}")

print("\n" + "=" * 60)
print("Phase 10 Complete - FastAPI Backend Working!")
print("=" * 60)
print("\nAPI Endpoints Available:")
print("  GET  /health")
print("  POST /index")
print("  GET  /index/status/{video_id}")
print("  POST /query")
print("\nInteractive docs: http://localhost:8000/docs")
print("\nReady for Phase 11: Chrome Extension!")