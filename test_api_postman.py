"""
Test script for the /audio/normalize/-14 endpoint
This script functions like a Postman request to test the API
"""
import requests
import os
import json

# API endpoint
BASE_URL = "http://localhost:8000"
TARGET_LUFS = "-14"  # Must be negative
ENDPOINT = f"{BASE_URL}/audio/normalize/{TARGET_LUFS}"

# Path to test audio file
current_dir = os.path.dirname(os.path.abspath(__file__))
example_audio_dir = os.path.join(current_dir, "example_audio")
sample_audio_path = os.path.join(example_audio_dir, "quiet_sine_440hz.wav")

# First verify that the file exists
print(f"Checking test file: {sample_audio_path}")
if not os.path.exists(sample_audio_path):
    available_files = os.listdir(example_audio_dir) if os.path.exists(example_audio_dir) else []
    print(f"ERROR: Test file not found. Available files: {available_files}")
    exit(1)

# Check file size and details
file_size = os.path.getsize(sample_audio_path)
print(f"File exists: {os.path.exists(sample_audio_path)}")
print(f"File size: {file_size} bytes")

# Step 1: Check if server is running
try:
    print("\nChecking server status...")
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        print(f"✅ Server is running. Status: {response.status_code}")
        print(f"Response: {response.json()}")
    else:
        print(f"⚠️ Server returned status {response.status_code}")
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error connecting to server: {e}")
    exit(1)

# Step 2: Check audio API status
try:
    print("\nChecking audio API status...")
    response = requests.get(f"{BASE_URL}/audio/status")
    if response.status_code == 200:
        print(f"✅ Audio API is working. Status: {response.status_code}")
        print(f"Response: {response.json()}")
    else:
        print(f"⚠️ Audio API returned status {response.status_code}")
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error checking audio API: {e}")

# Step 3: Check dependencies
try:
    print("\nChecking audio dependencies...")
    response = requests.get(f"{BASE_URL}/audio/dependencies")
    if response.status_code == 200:
        print(f"✅ Dependencies check passed. Status: {response.status_code}")
        deps = response.json().get("audio_dependencies", {})
        for dep, info in deps.items():
            status = "✅" if info.get("available", False) else "❌"
            version = info.get("version", "N/A")
            print(f"  {status} {dep}: {version}")
    else:
        print(f"⚠️ Dependencies check returned status {response.status_code}")
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error checking dependencies: {e}")

# Step 4: Test the normalize endpoint
print(f"\n🔄 Testing {ENDPOINT} with file {os.path.basename(sample_audio_path)}...")
try:
    with open(sample_audio_path, 'rb') as f:
        files = {'file': (os.path.basename(sample_audio_path), f, 'audio/wav')}
        
        print(f"Sending POST request to {ENDPOINT}")
        response = requests.post(ENDPOINT, files=files)
        
        # Print response details
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        try:
            # Try to parse as JSON
            result = response.json()
            print("\nResponse Body (JSON):")
            print(json.dumps(result, indent=2))
            
            # If successful, try to download the normalized file
            if response.status_code == 200 and "download_url" in result:
                download_url = result["download_url"]
                print(f"\n📥 Testing download from {download_url}")
                
                download_response = requests.get(download_url)
                print(f"Download Status: {download_response.status_code}")
                
                if download_response.status_code == 200:
                    content_length = download_response.headers.get("Content-Length", "unknown")
                    content_type = download_response.headers.get("Content-Type", "unknown")
                    print(f"✅ Download successful! Size: {content_length} bytes, Type: {content_type}")
                    
                    # Save the downloaded file
                    output_path = os.path.join(current_dir, f"downloaded_normalized_{TARGET_LUFS.replace('-', 'neg')}.wav")
                    with open(output_path, "wb") as f:
                        f.write(download_response.content)
                    print(f"✅ Saved normalized file to: {output_path}")
                else:
                    print(f"❌ Download failed: {download_response.text[:200]}")
        except ValueError:
            # Not JSON
            print("\nResponse Body (not JSON):")
            print(response.text[:1000])  # Print first 1000 chars
            
except Exception as e:
    print(f"❌ Error during API test: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test completed!")
