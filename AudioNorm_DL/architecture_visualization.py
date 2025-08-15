# 🎵 AudioNorm CNN Architecture Visualization
# =====================================================

"""
Your Custom CNN Architecture for Audio Normalization
===================================================

INPUT PROCESSING:
┌─────────────────────────────────────────────────┐
│  Raw Audio File (WAV/MP3/FLAC/etc.)           │
│  ↓                                             │
│  Load & Resample to 48kHz                     │
│  ↓                                             │
│  Extract Mel-Spectrogram (64 mel bins)        │
│  ↓                                             │
│  Extract Additional Features (9 features)      │
└─────────────────────────────────────────────────┘

DUAL INPUT STREAMS:
┌─────────────────────────────────────┐    ┌────────────────────────┐
│        SPECTROGRAM STREAM           │    │   FEATURE STREAM       │
│   Shape: (batch, 64, time_frames)   │    │   Shape: (batch, 9)    │
│                                     │    │                        │
│   Features:                         │    │   Features:            │
│   • Mel-frequency coefficients     │    │   • RMS Energy         │
│   • Time-frequency representation  │    │   • Peak Amplitude     │
│   • Power spectral density         │    │   • Crest Factor       │
│                                     │    │   • Spectral Centroid │
│                                     │    │   • Spectral Rolloff   │
│                                     │    │   • Zero Crossing Rate │
│                                     │    │   • Statistical Moments│
└─────────────────────────────────────┘    └────────────────────────┘
                    │                                    │
                    ▼                                    │
                                                         │
CNN PROCESSING PIPELINE:                                 │
                                                         │
┌───────────────────────────────────────────────────────┐│
│  INPUT: Add Channel Dimension                         ││
│  Shape: (batch, 1, 64, time_frames)                  ││
└───────────────────────────────────────────────────────┘│
                    │                                    │
                    ▼                                    │
┌───────────────────────────────────────────────────────┐│
│  CONV BLOCK 1                                         ││
│  ┌─────────────────────────────────────────────────┐  ││
│  │ Conv2d(1→32, 3×3, pad=1)                       │  ││
│  │ BatchNorm2d(32)                                 │  ││
│  │ ReLU()                                          │  ││
│  │ MaxPool2d(2×2)        ← Reduces spatial dims   │  ││
│  │ Dropout2d(0.25)                                 │  ││
│  └─────────────────────────────────────────────────┘  ││
│  Output: (batch, 32, 32, time_frames/2)               ││
└───────────────────────────────────────────────────────┘│
                    │                                    │
                    ▼                                    │
┌───────────────────────────────────────────────────────┐│
│  CONV BLOCK 2                                         ││
│  ┌─────────────────────────────────────────────────┐  ││
│  │ Conv2d(32→64, 3×3, pad=1)                       │  ││
│  │ BatchNorm2d(64)                                 │  ││
│  │ ReLU()                                          │  ││
│  │ MaxPool2d(2×2)        ← Further reduction      │  ││
│  │ Dropout2d(0.25)                                 │  ││
│  └─────────────────────────────────────────────────┘  ││
│  Output: (batch, 64, 16, time_frames/4)               ││
└───────────────────────────────────────────────────────┘│
                    │                                    │
                    ▼                                    │
┌───────────────────────────────────────────────────────┐│
│  CONV BLOCK 3                                         ││
│  ┌─────────────────────────────────────────────────┐  ││
│  │ Conv2d(64→128, 3×3, pad=1)                      │  ││
│  │ BatchNorm2d(128)                                │  ││
│  │ ReLU()                                          │  ││
│  │ AdaptiveAvgPool2d(4×4)  ← Fixed output size    │  ││
│  │ Dropout2d(0.25)                                 │  ││
│  └─────────────────────────────────────────────────┘  ││
│  Output: (batch, 128, 4, 4)                           ││
└───────────────────────────────────────────────────────┘│
                    │                                    │
                    ▼                                    │
┌───────────────────────────────────────────────────────┐│
│  FLATTEN                                              ││
│  Shape: (batch, 128×4×4) = (batch, 2048)             ││
└───────────────────────────────────────────────────────┘│
                    │                                    │
                    ▼                                    │
                                                         │
FEATURE FUSION:                                          │
┌─────────────────────────────────────────────────────────┐
│  CONCATENATE FEATURES                                   │
│  CNN Features: (batch, 2048)                          │
│  Additional Features: (batch, 9)              ◄───────┘
│  Combined: (batch, 2057)                               │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
FULLY CONNECTED PIPELINE:
┌─────────────────────────────────────────────────────────┐
│  FC BLOCK 1                                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Linear(2057 → 256)                                │  │
│  │ BatchNorm1d(256)                                  │  │
│  │ ReLU()                                            │  │
│  │ Dropout(0.5)        ← High dropout for regulariz │  │
│  └───────────────────────────────────────────────────┘  │
│  Output: (batch, 256)                                   │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  FC BLOCK 2                                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Linear(256 → 128)                                 │  │
│  │ BatchNorm1d(128)                                  │  │
│  │ ReLU()                                            │  │
│  │ Dropout(0.3)        ← Medium dropout             │  │
│  └───────────────────────────────────────────────────┘  │
│  Output: (batch, 128)                                   │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  FC BLOCK 3                                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Linear(128 → 64)                                  │  │
│  │ ReLU()                                            │  │
│  │ Dropout(0.2)        ← Low dropout                │  │
│  └───────────────────────────────────────────────────┘  │
│  Output: (batch, 64)                                    │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  OUTPUT LAYER                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Linear(64 → 1)                                    │  │
│  │ No activation (regression)                        │  │
│  └───────────────────────────────────────────────────┘  │
│  Output: (batch, 1) = Predicted Gain in dB             │
└─────────────────────────────────────────────────────────┘

ARCHITECTURE SUMMARY:
=====================
📊 Input Dimensions:
   • Spectrogram: Variable time × 64 mel bins
   • Additional Features: 9 statistical features

🧠 Network Structure:
   • 3 Convolutional Blocks (32→64→128 filters)
   • 4 Fully Connected Layers (2057→256→128→64→1)
   • Regularization: BatchNorm + Dropout at every layer

🎯 Output:
   • Single regression value: Gain adjustment in dB
   • Range: Typically -30dB to +30dB

⚡ Key Features:
   • Dual-stream processing (CNN + handcrafted features)
   • Adaptive pooling for variable input sizes
   • Progressive dimensionality reduction
   • Heavy regularization to prevent overfitting

📈 Parameter Count Estimate:
   • Conv layers: ~150K parameters
   • FC layers: ~600K parameters  
   • Total: ~750K parameters (lightweight!)

🔧 Training Strategy:
   • Loss: L1 Loss (Mean Absolute Error)
   • Optimizer: AdamW with weight decay
   • Target LUFS: -10, -12, -14 dB
   • Data augmentation via random audio crops
"""

# ASCII Architecture Diagram
ARCHITECTURE_ASCII = """
Audio File → Spectrogram (64×T) ──┐
                                  │
Additional Features (9) ──────────┼─→ Concat ─→ FC Layers ─→ Gain (dB)
                                  │    ↑
                Conv1(32) ────────┤    │
                    ↓             │    │
                Conv2(64) ────────┤    │
                    ↓             │    │
                Conv3(128) ───────┤    │
                    ↓             │    │
                Flatten ──────────┘    │
                    ↓                  │
                (2048 features) ───────┘

Legend:
• Conv blocks include: Conv2d → BatchNorm → ReLU → Pool → Dropout
• FC blocks include: Linear → BatchNorm → ReLU → Dropout  
• Final layer: Linear(64→1) with no activation
"""

print("🎵 Your Custom AudioNorm CNN Architecture")
print("=" * 50)
print(ARCHITECTURE_ASCII)
