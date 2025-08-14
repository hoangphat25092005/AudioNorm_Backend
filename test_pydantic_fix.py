import requests
import json

BASE_URL = "http://localhost:8002"

def test_fix():
    """Test that the Pydantic v2 fix worked"""
    try:
        print("Testing dependencies endpoint...")
        response = requests.get(f"{BASE_URL}/audio/dependencies")
        print(f"Dependencies Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Service type: {data.get('service_type', 'unknown')}")
            print("✅ Dependencies endpoint working!")
            
            # Test normalize with a simple request
            print("\nTesting normalize endpoint...")
            
            # Create minimal test data
            test_data = b"test audio data"
            files = {'file': ('test.wav', test_data, 'audio/wav')}
            
            response = requests.post(f"{BASE_URL}/audio/normalize/-14", files=files)
            print(f"Normalize Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Normalize endpoint working!")
                print(f"Service type used: {result.get('service_type', 'unknown')}")
                if result.get('warning'):
                    print(f"Warning: {result['warning']}")
            else:
                print("❌ Normalize endpoint failed")
                try:
                    error = response.json()
                    print(f"Error: {error}")
                except:
                    print(f"Raw error: {response.text[:200]}")
        else:
            print("❌ Dependencies endpoint failed")
            print(f"Error: {response.text[:200]}")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_fix()
