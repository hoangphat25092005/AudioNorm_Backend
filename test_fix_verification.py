import requests
import json
import os

BASE_URL = "http://localhost:8001"

def test_dependencies():
    """Test the dependencies endpoint"""
    try:
        print("Testing dependencies endpoint...")
        response = requests.get(f"{BASE_URL}/audio/dependencies")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Dependencies check result:")
            print(json.dumps(data, indent=2))
            return data
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_normalize():
    """Test the normalize endpoint with a minimal file"""
    try:
        print("\nTesting normalize endpoint...")
        
        # Create a minimal WAV file for testing
        wav_header = bytes([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x2C, 0x00, 0x00, 0x00,  # File size (44 bytes)
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # Subchunk1Size (16)
            0x01, 0x00,              # AudioFormat (PCM)
            0x01, 0x00,              # NumChannels (1)
            0x44, 0xAC, 0x00, 0x00,  # SampleRate (44100)
            0x88, 0x58, 0x01, 0x00,  # ByteRate
            0x02, 0x00,              # BlockAlign
            0x10, 0x00,              # BitsPerSample (16)
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x08, 0x00, 0x00, 0x00,  # Subchunk2Size (8 bytes of data)
            0x00, 0x00, 0x00, 0x00,  # 4 bytes of silence
            0x00, 0x00, 0x00, 0x00   # 4 more bytes of silence
        ])
        
        # Save test file
        test_file_path = "test_minimal.wav"
        with open(test_file_path, "wb") as f:
            f.write(wav_header)
        
        print(f"Created test file: {test_file_path} ({len(wav_header)} bytes)")
        
        # Test the normalize endpoint
        url = f"{BASE_URL}/audio/normalize/-14"
        with open(test_file_path, 'rb') as f:
            files = {'file': (test_file_path, f, 'audio/wav')}
            response = requests.post(url, files=files)
        
        print(f"Normalize Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Normalize result:")
            print(json.dumps(data, indent=2))
            
            # Test download
            if "download_url" in data:
                print(f"\nTesting download from: {data['download_url']}")
                download_response = requests.get(data['download_url'])
                print(f"Download Status: {download_response.status_code}")
                if download_response.status_code == 200:
                    print(f"✅ Download successful! Content-Type: {download_response.headers.get('content-type')}")
                else:
                    print(f"❌ Download failed: {download_response.text}")
        else:
            try:
                error_data = response.json()
                print("Error response:")
                print(json.dumps(error_data, indent=2))
            except:
                print(f"Raw error response: {response.text}")
        
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            
    except Exception as e:
        print(f"Error in normalize test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing AudioNorm API on port 8001...")
    test_dependencies()
    test_normalize()
    print("\nTest completed!")
