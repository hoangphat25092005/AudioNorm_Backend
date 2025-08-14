"""
Create a simple sine wave audio file for testing the API
"""
import os
import numpy as np
import soundfile as sf

# Create the example_audio directory if it doesn't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
example_audio_dir = os.path.join(current_dir, "example_audio")
os.makedirs(example_audio_dir, exist_ok=True)

# Generate a quiet sine wave at 440Hz
sample_rate = 44100  # Hz
duration = 3.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
frequency = 440.0  # Hz (A4 note)
amplitude = 0.1  # Quiet - should be around -20 dB LUFS

# Generate the sine wave
quiet_sine = amplitude * np.sin(2 * np.pi * frequency * t)

# Save as WAV file
output_path = os.path.join(example_audio_dir, "quiet_sine_440hz.wav")
sf.write(output_path, quiet_sine, sample_rate)
print(f"Created test audio file: {output_path}")
print(f"File exists: {os.path.exists(output_path)}")
print(f"File size: {os.path.getsize(output_path)} bytes")

# Generate a louder sine wave at 880Hz (one octave higher)
frequency = 880.0  # Hz (A5 note)
amplitude = 0.8  # Loud - should be around -3 dB LUFS

# Generate the sine wave
loud_sine = amplitude * np.sin(2 * np.pi * frequency * t)

# Save as WAV file
output_path = os.path.join(example_audio_dir, "loud_sine_880hz.wav")
sf.write(output_path, loud_sine, sample_rate)
print(f"Created test audio file: {output_path}")

# Create a very quiet noise sample
noise = np.random.normal(0, 0.01, int(sample_rate * duration))
output_path = os.path.join(example_audio_dir, "quiet_noise.wav")
sf.write(output_path, noise, sample_rate)
print(f"Created test audio file: {output_path}")

# Create a simple C major chord
c_freq = 261.63  # C4
e_freq = 329.63  # E4
g_freq = 392.00  # G4

c_note = 0.3 * np.sin(2 * np.pi * c_freq * t)
e_note = 0.3 * np.sin(2 * np.pi * e_freq * t)
g_note = 0.3 * np.sin(2 * np.pi * g_freq * t)

c_major = c_note + e_note + g_note

output_path = os.path.join(example_audio_dir, "c_major_chord.wav")
sf.write(output_path, c_major, sample_rate)
print(f"Created test audio file: {output_path}")

print("\nAll test audio files created successfully!")
