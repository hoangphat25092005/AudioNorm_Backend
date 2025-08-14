"""Check if all required dependencies are installed"""
import sys

print("Checking Python version:")
print(f"Python {sys.version}")

print("\nChecking required dependencies:")

# Check FastAPI and basic dependencies
try:
    import fastapi
    print(f"✅ fastapi: {fastapi.__version__}")
except ImportError:
    print("❌ fastapi: NOT INSTALLED")

try:
    import uvicorn
    print(f"✅ uvicorn: {uvicorn.__version__}")
except ImportError:
    print("❌ uvicorn: NOT INSTALLED")

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    print("✅ motor: INSTALLED")
except ImportError:
    print("❌ motor: NOT INSTALLED")

# Check audio processing dependencies
try:
    import numpy as np
    print(f"✅ numpy: {np.__version__}")
except ImportError:
    print("❌ numpy: NOT INSTALLED")

try:
    import librosa
    print(f"✅ librosa: {librosa.__version__}")
except ImportError:
    print("❌ librosa: NOT INSTALLED - REQUIRED FOR AUDIO PROCESSING")

try:
    import pyloudnorm as pyln
    print(f"✅ pyloudnorm: {pyln.__version__}")
except ImportError:
    print("❌ pyloudnorm: NOT INSTALLED - REQUIRED FOR LUFS MEASUREMENT")

try:
    import soundfile as sf
    print(f"✅ soundfile: {sf.__version__}")
except ImportError:
    print("❌ soundfile: NOT INSTALLED - REQUIRED FOR AUDIO I/O")

try:
    import torch
    print(f"✅ torch: {torch.__version__}")
except ImportError:
    print("❌ torch: NOT INSTALLED - REQUIRED FOR DL MODEL")

# Test basic functionality
print("\nTesting basic imports from our modules:")
try:
    from app.config.database import get_db
    print("✅ Database config: OK")
except Exception as e:
    print(f"❌ Database config: {e}")

try:
    from app.models.audio_model import AudioNormalizationResult
    print("✅ Audio model: OK")
except Exception as e:
    print(f"❌ Audio model: {e}")

try:
    from app.services.audio_service import AudioService
    print("✅ Audio service import: OK")
except Exception as e:
    print(f"❌ Audio service import: {e}")

print("\nDependency check completed!")

# If missing critical dependencies, show install command
missing_audio_deps = []
for dep in ['librosa', 'pyloudnorm', 'soundfile', 'torch']:
    try:
        __import__(dep)
    except ImportError:
        missing_audio_deps.append(dep)

if missing_audio_deps:
    print(f"\n⚠️ Missing audio dependencies: {', '.join(missing_audio_deps)}")
    print("To install missing dependencies, run:")
    print(f"pip install {' '.join(missing_audio_deps)}")
    
    # Special note for torch
    if 'torch' in missing_audio_deps:
        print("\nFor torch, you might want to use the CPU-only version:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
