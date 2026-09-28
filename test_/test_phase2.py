"""
Quick test for Phase 2
"""

from indexing.document_loader import load_youtube_transcript
from utils.validators import extract_video_id_from_url

print("=" * 60)
print("Phase 2 Verification")
print("=" * 60)

# Test 1: Extract ID from URL
url = "https://www.youtube.com/watch?v=kY14KfZQ1TI"
video_id = extract_video_id_from_url(url)
print(f"\nExtracted ID: {video_id}")

# Test 2: Load transcript
try:
    result = load_youtube_transcript(video_id)
    
    print(f"\nSuccess!")
    print(f"  Segments: {result['num_segments']}")
    print(f"  Characters: {result['total_chars']}")
    print(f"  Preview: {result['text'][:150]}...")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)