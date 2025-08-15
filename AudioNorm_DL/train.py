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
MODEL_OUT = "models/norm_cnn.pth"
TARGET_TIME_FRAMES = 128  # Fixed time dimension for CNN

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
    # Extract mel-spectrogram for CNN processing
    # Use shorter hop length for better time resolution
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
    
    # Return as (n_mels, time_frames) for CNN
    return S_db.astype(np.float32)

def extract_additional_features(y, sr):
    # Extract additional scalar features for concatenation
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
        
        # Extract spectrogram and additional features
        spectrogram = extract_features(y, sr)  # (n_mels, time_frames)
        additional_features = extract_additional_features(y, sr)  # (9,)
        
        target = random.choice(TARGETS_LUFS)
        true_gain_db = target - lufs   # ideal single-step gain to hit target
        # Clamp label to a reasonable range to avoid extreme outliers
        true_gain_db = float(np.clip(true_gain_db, -30.0, 30.0))
        
        return spectrogram, additional_features, np.array([true_gain_db], dtype=np.float32)

# ----- Model -----
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

# ----- Train -----
def train_one_epoch(model, loader, opt):
    model.train()
    crit = nn.L1Loss()
    epoch_loss = 0.0
    for spectrograms, additional_feats, gain_db in tqdm(loader, desc="train", leave=False):
        spectrograms = spectrograms.to(DEVICE)
        additional_feats = additional_feats.to(DEVICE)
        gain_db = gain_db.to(DEVICE)
        
        pred = model(spectrograms, additional_feats)
        loss = crit(pred, gain_db)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * spectrograms.size(0)
    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def eval_loss(model, loader):
    model.eval()
    crit = nn.L1Loss()
    epoch_loss = 0.0
    for spectrograms, additional_feats, gain_db in loader:
        spectrograms = spectrograms.to(DEVICE)
        additional_feats = additional_feats.to(DEVICE)
        gain_db = gain_db.to(DEVICE)
        
        pred = model(spectrograms, additional_feats)
        loss = crit(pred, gain_db)
        epoch_loss += loss.item() * spectrograms.size(0)
    return epoch_loss / len(loader.dataset)

def main():
    os.makedirs("models", exist_ok=True)
    train_set = GainDataset("data/train")
    val_dir = "data/val" if os.path.isdir("data/val") else None
    val_set = GainDataset(val_dir) if val_dir else None

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)  # Reduced batch size for CNN
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0) if val_set else None

    model = AudioNormCNN(n_mels=N_MELS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Lower learning rate for CNN
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.7, patience=3)

    best = float("inf")
    EPOCHS = 50  # More epochs for CNN training
    for epoch in range(1, EPOCHS+1):
        tr = train_one_epoch(model, train_loader, opt)
        if val_loader:
            va = eval_loss(model, val_loader)
            print(f"Epoch {epoch}: train L1={tr:.3f} dB, val L1={va:.3f} dB, lr={opt.param_groups[0]['lr']:.2e}")
            metric = va
            scheduler.step(va)
        else:
            print(f"Epoch {epoch}: train L1={tr:.3f} dB, lr={opt.param_groups[0]['lr']:.2e}")
            metric = tr
            scheduler.step(tr)
            
        if metric < best:
            best = metric
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  saved -> {MODEL_OUT} (best {best:.3f} dB)")

    if not os.path.exists(MODEL_OUT):
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"Saved final model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
