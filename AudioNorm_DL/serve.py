import os, tempfile, subprocess
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse
import numpy as np
import librosa
import pyloudnorm as pyln
import torch
import torch.nn as nn

# --- Config ---
SR = 48000
N_MELS = 64
MODEL_PATH = "models/norm_cnn.pth"
TARGET_TIME_FRAMES = 128  # Fixed time dimension for CNN
REFINE_EXACT = True    # set False to skip post-refinement
TMP_DIR = None         # None → system temp

app = FastAPI(title="DL Audio Normalization API", version="1.0.0")

# --- Features (must match training) ---
def extract_features(y, sr, n_mels=N_MELS):
    # Extract mel-spectrogram for CNN processing
    hop_length = 512
    n_fft = 2048
    
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, 
        hop_length=hop_length, power=2.0
    )
    S_db = librosa.power_to_db(S + 1e-12)
    
    # Normalize the spectrogram
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
    
    # Resize to fixed time dimension
    if S_db.shape[1] < TARGET_TIME_FRAMES:
        # Pad with zeros if too short
        pad_width = TARGET_TIME_FRAMES - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
    elif S_db.shape[1] > TARGET_TIME_FRAMES:
        # Crop if too long (take center portion)
        start = (S_db.shape[1] - TARGET_TIME_FRAMES) // 2
        S_db = S_db[:, start:start + TARGET_TIME_FRAMES]
    
    return S_db.astype(np.float32)

def extract_additional_features(y, sr):
    # Extract additional scalar features
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    peak = float(np.max(np.abs(y)) + 1e-12)
    crest = float(peak / (rms + 1e-12))
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Aggregate temporal features
    features = np.array([
        rms, peak, crest,
        np.mean(spectral_centroids), np.std(spectral_centroids),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
    ], dtype=np.float32)
    
    return features

def load_mono_tempfile(upload: UploadFile, sr=SR) -> str:
    # Save upload to a temp file, then convert to wav mono SR using librosa for consistency
    suffix = os.path.splitext(upload.filename or "in")[1] or ".bin"
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP_DIR)
    tmp_in.write(upload.file.read())
    tmp_in.close()
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TMP_DIR).name
    
    # Use librosa instead of ffmpeg
    try:
        y, _ = librosa.load(tmp_in.name, sr=sr, mono=True)
        import soundfile as sf
        sf.write(tmp_wav, y, sr)
    except Exception as e:
        print(f"Error converting audio: {e}")
        raise
    
    os.unlink(tmp_in.name)
    return tmp_wav

def measure_lufs_file(path: str) -> float:
    y, sr = librosa.load(path, sr=SR, mono=True)
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(y))

def apply_gain_ffmpeg(in_path: str, out_path: str, gain_db: float):
    # Use librosa and soundfile to apply precise gain in dB
    try:
        y, sr = librosa.load(in_path, sr=SR, mono=True)
        # Convert dB gain to amplitude multiplier
        gain_factor = 10 ** (gain_db / 20.0)
        # Apply gain
        y_gained = y * gain_factor
        
        # Save as wav
        import soundfile as sf
        sf.write(out_path, y_gained, sr)
    except Exception as e:
        print(f"Error applying gain: {e}")
        raise

# --- Model must mirror training ---
class AudioNormCNN(nn.Module):
    def __init__(self, n_mels=N_MELS, additional_features_dim=9):
        super().__init__()
        
        # CNN for processing spectrograms
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed output size
            nn.Dropout2d(0.25),
        )
        
        # Calculate flattened CNN output size
        self.cnn_output_size = 128 * 4 * 4  # 128 channels * 4 * 4
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.cnn_output_size + additional_features_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)  # Output: gain in dB
        )
        
    def forward(self, spectrogram, additional_features):
        # spectrogram: (batch, n_mels, time_frames)
        # additional_features: (batch, 9)
        
        # Add channel dimension for CNN: (batch, 1, n_mels, time_frames)
        x = spectrogram.unsqueeze(1)
        
        # Process through CNN
        x = self.conv_layers(x)
        
        # Flatten CNN output
        x = x.view(x.size(0), -1)  # (batch, cnn_output_size)
        
        # Concatenate with additional features
        x = torch.cat([x, additional_features], dim=1)
        
        # Process through fully connected layers
        x = self.fc_layers(x)
        
        return x

def feature_dim(): 
    # Not used for CNN, but keeping for compatibility
    return 2*N_MELS + 3

# Load model once
MODEL: Optional[AudioNormCNN] = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_model():
    global MODEL
    MODEL = AudioNormCNN(n_mels=N_MELS).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        MODEL.eval()
        print(f"Loaded model: {MODEL_PATH}")
    else:
        print(f"WARNING: {MODEL_PATH} not found. The API will still run but predictions may be random.")

def predict_gain_db(y, sr, target_lufs):
    spectrogram = extract_features(y, sr)
    additional_features = extract_additional_features(y, sr)
    
    # Convert to tensors and add batch dimension
    spec_tensor = torch.from_numpy(spectrogram[None, :, :]).to(DEVICE)  # (1, n_mels, time_frames)
    feat_tensor = torch.from_numpy(additional_features[None, :]).to(DEVICE)  # (1, 9)
    
    with torch.no_grad():
        pred = MODEL(spec_tensor, feat_tensor).cpu().numpy()[0, 0]
    return float(pred)

def refine_exact_lufs(in_path: str, target_lufs: float) -> str:
    """
    Optional precise snap-to-target:
    Measure -> apply residual gain once more.
    """
    measured = measure_lufs_file(in_path)
    residual = target_lufs - measured
    if abs(residual) < 0.1:
        return in_path
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TMP_DIR).name
    apply_gain_ffmpeg(in_path, tmp_out, residual)
    os.unlink(in_path)
    return tmp_out

def normalize_with_model(in_wav_path: str, target_lufs: float) -> str:
    # Load waveform for features
    y, sr = librosa.load(in_wav_path, sr=SR, mono=True)
    # 1) DL predicts initial gain
    pred_gain = predict_gain_db(y, sr, target_lufs)
    # 2) Shift initial gain toward requested LUFS (add a small bias)
    #    We know ideal = target - measured, so bias by the sign/diff:
    measured = measure_lufs_file(in_wav_path)
    ideal = target_lufs - measured
    blended = 0.7 * pred_gain + 0.3 * ideal   # small stabilizer
    # 3) Apply predicted gain
    tmp_pred = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TMP_DIR).name
    apply_gain_ffmpeg(in_wav_path, tmp_pred, blended)
    # 4) Optional exact snap
    if REFINE_EXACT:
        tmp_pred = refine_exact_lufs(tmp_pred, target_lufs)
    return tmp_pred

def to_download_name(target_lufs: float, orig_name: Optional[str]) -> str:
    base = (orig_name or "audio").rsplit(".", 1)[0]
    level = int(abs(target_lufs))
    return f"{base}_norm_{level}LUFS.wav"

# --------- Routes ---------

@app.post("/normalize/")
async def normalize_generic(
    file: UploadFile = File(...),
    target_lufs: float = Query(..., description="Target LUFS, e.g., -10, -12, -14")
):
    wav_in = load_mono_tempfile(file, sr=SR)
    wav_out = normalize_with_model(wav_in, target_lufs)
    filename = to_download_name(target_lufs, file.filename)
    return FileResponse(wav_out, media_type="audio/wav", filename=filename)

@app.post("/normalize/10")
async def normalize_10(file: UploadFile = File(...)):
    return await normalize_generic(file=file, target_lufs=-10.0)

@app.post("/normalize/12")
async def normalize_12(file: UploadFile = File(...)):
    return await normalize_generic(file=file, target_lufs=-12.0)

@app.post("/normalize/14")
async def normalize_14(file: UploadFile = File(...)):
    return await normalize_generic(file=file, target_lufs=-14.0)
