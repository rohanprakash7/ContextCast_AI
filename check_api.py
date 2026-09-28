# check_api.py
from youtube_transcript_api import YouTubeTranscriptApi

print("Available methods in YouTubeTranscriptApi:")
for method in dir(YouTubeTranscriptApi):
    if not method.startswith('_'):
        print(f"  - {method}")

print("\n" + "="*60)
print("Trying to get transcript for kY14KfZQ1TI...")

video_id = "kY14KfZQ1TI"

# Try different approaches
try:
    # Approach 1: Instance method
    api = YouTubeTranscriptApi()
    transcript = api.get_transcript(video_id)
    print("SUCCESS with instance method!")
    print(f"Got {len(transcript)} segments")
except Exception as e:
    print(f"Instance method failed: {e}")

try:
    # Approach 2: Check if it's a module-level function
    from youtube_transcript_api import get_transcript
    transcript = get_transcript(video_id)
    print("SUCCESS with direct import!")
    print(f"Got {len(transcript)} segments")
except Exception as e:
    print(f"Direct import failed: {e}")