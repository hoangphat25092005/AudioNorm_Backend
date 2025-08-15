#!/usr/bin/env python3
"""
Test script for CNN-based audio normalization training
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add the parent directory to path to import our modules
sys.path.append(os.path.dirname(__file__))

from train import AudioNormCNN, extract_features, extract_additional_features
import librosa

def test_model_architecture():
    """Test if the CNN model can be instantiated and forward pass works"""
    print("Testing CNN model architecture...")
    
    # Model parameters
    n_mels = 64
    additional_features_dim = 9
    target_time_frames = 128
    
    # Create model
    model = AudioNormCNN(n_mels=n_mels, additional_features_dim=additional_features_dim)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    spectrogram = torch.randn(batch_size, n_mels, target_time_frames)
    additional_features = torch.randn(batch_size, additional_features_dim)
    
    # Forward pass
    with torch.no_grad():
        output = model(spectrogram, additional_features)
    
    print(f"✅ Model forward pass successful!")
    print(f"   Input spectrogram shape: {spectrogram.shape}")
    print(f"   Input additional features shape: {additional_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return True

def test_feature_extraction():
    """Test feature extraction pipeline"""
    print("\nTesting feature extraction...")
    
    # Create a dummy audio signal (5 seconds of sine wave)
    sr = 48000
    duration = 5.0
    freq = 440.0  # A4 note
    t = np.linspace(0, duration, int(sr * duration), False)
    y = 0.1 * np.sin(freq * 2 * np.pi * t)
    
    # Extract features
    spectrogram = extract_features(y, sr)
    additional_features = extract_additional_features(y, sr)
    
    print(f"✅ Feature extraction successful!")
    print(f"   Audio duration: {duration}s, sample rate: {sr}Hz")
    print(f"   Spectrogram shape: {spectrogram.shape}")
    print(f"   Additional features shape: {additional_features.shape}")
    print(f"   Spectrogram range: [{spectrogram.min():.3f}, {spectrogram.max():.3f}]")
    print(f"   Additional features: {additional_features}")
    
    return spectrogram, additional_features

def test_training_pipeline():
    """Test a mini training step"""
    print("\nTesting training pipeline...")
    
    # Create model
    model = AudioNormCNN()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy batch
    batch_size = 4
    n_mels = 64
    target_time_frames = 128
    
    spectrograms = torch.randn(batch_size, n_mels, target_time_frames)
    additional_features = torch.randn(batch_size, 9)
    targets = torch.randn(batch_size, 1) * 10  # Random gain targets
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    outputs = model(spectrograms, additional_features)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print(f"✅ Training step successful!")
    print(f"   Batch size: {batch_size}")
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Output mean: {outputs.mean().item():.3f}")
    print(f"   Target mean: {targets.mean().item():.3f}")
    
    return True

def main():
    """Run all tests"""
    print("🎵 Testing CNN Audio Normalization Model 🎵")
    print("=" * 50)
    
    try:
        # Test 1: Model architecture
        test_model_architecture()
        
        # Test 2: Feature extraction
        spectrogram, additional_features = test_feature_extraction()
        
        # Test 3: Training pipeline
        test_training_pipeline()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! The CNN model is ready for training.")
        print("\nTo start training:")
        print("1. Prepare your audio dataset in 'data/train/' folder")
        print("2. Run: python train.py")
        print("3. The trained model will be saved as 'models/norm_cnn.pth'")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
