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
MODEL_PATH = "models/norm_mlp.pth"
REFINE_EXACT = True    # set False to skip post-refinement
TMP_DIR = None         # None → system temp

app = FastAPI(title="DL Audio Normalization API", version="1.0.0")

# --- Features (must match training) ---
def extract_features(y, sr, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S + 1e-12)
    mean = S_db.mean(axis=1)
    std  = S_db.std(axis=1)
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    peak = float(np.max(np.abs(y)) + 1e-12)
    crest = float(peak / (rms + 1e-12))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    feat = np.concatenate([mean, std, np.array([rms, crest, flatness], dtype=np.float32)])
    return feat.astype(np.float32)

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
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

def feature_dim(): return 2*N_MELS + 3

# Load model once
MODEL: Optional[MLP] = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_model():
    global MODEL
    MODEL = MLP(feature_dim()).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        MODEL.eval()
        print(f"Loaded model: {MODEL_PATH}")
    else:
        print(f"WARNING: {MODEL_PATH} not found. The API will still run but predictions may be random.")

def predict_gain_db(y, sr, target_lufs):
    feat = extract_features(y, sr)
    x = torch.from_numpy(feat[None, :]).to(DEVICE)
    with torch.no_grad():
        pred = MODEL(x).cpu().numpy()[0,0]
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
