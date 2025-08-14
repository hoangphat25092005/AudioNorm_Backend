import os, random, math, glob
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----- Config -----
SR = 48000
N_MELS = 64
TARGETS_LUFS = [-10.0, -12.0, -14.0]  # you can add more
MAX_SECONDS = 30.0   # trim/segment to speed up feature extraction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OUT = "models/norm_mlp.pth"

# ----- Features -----
def load_mono(path, sr=SR, max_seconds=MAX_SECONDS):
    # Load mono float32, optionally trim to max_seconds for speed/consistency
    y, sr = librosa.load(path, sr=sr, mono=True)
    if max_seconds is not None:
        max_len = int(sr * max_seconds)
        if len(y) > max_len:
            # random 30s crop if long
            start = random.randint(0, len(y) - max_len)
            y = y[start:start+max_len]
    return y, sr

def measure_lufs(y, sr):
    # BS.1770 integrated loudness (mono)
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(y))

def extract_features(y, sr, n_mels=N_MELS):
    # Log-mel spectrogram → global statistics
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S + 1e-12)
    mean = S_db.mean(axis=1)
    std  = S_db.std(axis=1)

    # Simple loudness-related features
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    peak = float(np.max(np.abs(y)) + 1e-12)
    crest = float(peak / (rms + 1e-12))

    # Spectral flatness (mean)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    feat = np.concatenate([mean, std, np.array([rms, crest, flatness], dtype=np.float32)])
    return feat.astype(np.float32)

# ----- Dataset -----
class GainDataset(Dataset):
    def __init__(self, root):
        exts = ("*.wav","*.mp3","*.m4a","*.flac","*.ogg")
        files = []
        for e in exts:
            files += glob.glob(os.path.join(root, e))
        self.files = files
        if not self.files:
            raise RuntimeError(f"No audio files found in {root}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        y, sr = load_mono(path)
        lufs = measure_lufs(y, sr)
        feat = extract_features(y, sr)

        target = random.choice(TARGETS_LUFS)
        true_gain_db = target - lufs   # ideal single-step gain to hit target
        # Clamp label to a reasonable range to avoid extreme outliers
        true_gain_db = float(np.clip(true_gain_db, -30.0, 30.0))
        return feat, np.array([true_gain_db], dtype=np.float32)

# ----- Model -----
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)   # predict gain in dB
        )
    def forward(self, x): return self.net(x)

def feature_dim():
    # mean(N_MELS) + std(N_MELS) + [rms, crest, flatness] = 2*N_MELS + 3
    return 2*N_MELS + 3

# ----- Train -----
def train_one_epoch(model, loader, opt):
    model.train()
    crit = nn.L1Loss()
    epoch_loss = 0.0
    for feats, gain_db in tqdm(loader, desc="train", leave=False):
        feats = feats.to(DEVICE)
        gain_db = gain_db.to(DEVICE)
        pred = model(feats)
        loss = crit(pred, gain_db)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * feats.size(0)
    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def eval_loss(model, loader):
    model.eval()
    crit = nn.L1Loss()
    epoch_loss = 0.0
    for feats, gain_db in loader:
        feats = feats.to(DEVICE)
        gain_db = gain_db.to(DEVICE)
        pred = model(feats)
        loss = crit(pred, gain_db)
        epoch_loss += loss.item() * feats.size(0)
    return epoch_loss / len(loader.dataset)

def main():
    os.makedirs("models", exist_ok=True)
    train_set = GainDataset("data/train")
    val_dir = "data/val" if os.path.isdir("data/val") else None
    val_set = GainDataset(val_dir) if val_dir else None

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0) if val_set else None

    model = MLP(feature_dim()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best = float("inf")
    EPOCHS = 25  # Changed from 10 to 5 epochs
    for epoch in range(1, EPOCHS+1):
        tr = train_one_epoch(model, train_loader, opt)
        if val_loader:
            va = eval_loss(model, val_loader)
            print(f"Epoch {epoch}: train L1={tr:.3f} dB, val L1={va:.3f} dB")
            metric = va
        else:
            print(f"Epoch {epoch}: train L1={tr:.3f} dB")
            metric = tr
        if metric < best:
            best = metric
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  saved -> {MODEL_OUT} (best {best:.3f} dB)")

    if not os.path.exists(MODEL_OUT):
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"Saved final model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
