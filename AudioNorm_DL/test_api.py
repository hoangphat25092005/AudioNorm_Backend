import requests
import os

# Path to a sample audio file
sample_file_path = os.path.join("..", "example_audio", "quiet_sine_440hz.wav")

if os.path.exists(sample_file_path):
    print(f"Using audio file: {sample_file_path}")
    with open(sample_file_path, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        
        # Test /normalize/10 endpoint
        url = "http://127.0.0.1:8000/normalize/10"
        print(f"Sending request to {url}")
        response = requests.post(url, files=files)
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            # Save the normalized audio
            output_path = "normalized_10lufs.wav"
            with open(output_path, "wb") as out_file:
                out_file.write(response.content)
            print(f"Normalized audio saved to {output_path}")
        else:
            print(f"Error response: {response.text}")
else:
    print(f"Sample file not found: {sample_file_path}")
    print("Looking for alternative audio files...")
    
    # List all available example audio files
    example_dir = os.path.join("..", "example_audio")
    if os.path.exists(example_dir):
        audio_files = [f for f in os.listdir(example_dir) if f.endswith('.wav')]
        if audio_files:
            print(f"Found these audio files: {audio_files}")
            print(f"Please modify this script to use one of these files.")
        else:
            print(f"No WAV files found in {example_dir}")
    else:
        print(f"Example audio directory not found: {example_dir}")
