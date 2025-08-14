"""
Audio Normalization Service
Handles audio processing business logic and database operations
"""
import os
import sys
import tempfile
import subprocess
import time
import uuid
import shutil
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, UploadFile
import numpy as np
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection

from app.config.database import get_db
from app.models.audio_model import AudioNormalizationResult, AudioAnalysisResult

# Add AudioNorm_DL to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
audionorm_dl_path = os.path.join(current_dir, "AudioNorm_DL")
if audionorm_dl_path not in sys.path and os.path.exists(audionorm_dl_path):
    sys.path.append(audionorm_dl_path)

# File storage path for normalized audio
STORAGE_PATH = os.path.join(current_dir, "storage", "normalized_audio")
os.makedirs(STORAGE_PATH, exist_ok=True)
print(f"Storage path initialized: {STORAGE_PATH}")
print(f"Storage path exists: {os.path.exists(STORAGE_PATH)}")
print(f"Storage path is writable: {os.access(STORAGE_PATH, os.W_OK)}")

# Optional imports - will be handled gracefully if not available
try:
    import librosa
    import pyloudnorm as pyln
    import torch
    import torch.nn as nn
    import soundfile as sf
    AUDIO_DEPS_AVAILABLE = True
    print("✅ Audio processing dependencies loaded successfully")
except ImportError as e:
    AUDIO_DEPS_AVAILABLE = False
    print(f"⚠️ Audio processing dependencies not available: {e}")

class AudioService:
    def __init__(self):
        self.db = None
        self.collection: Optional[AsyncIOMotorCollection] = None
        self.model = None
        self._load_model()
    
    async def _get_collection(self):
        """Get the audio normalization collection"""
        if self.collection is None:
            self.db = await get_db()
            self.collection = self.db.audio_normalizations
        return self.collection
    
    def _load_model(self):
        """Load the pre-trained DL model if available"""
        if not AUDIO_DEPS_AVAILABLE:
            return
            
        try:
            # Updated MLP model matching the one from AudioNorm_DL/train.py
            class MLP(nn.Module):
                def __init__(self, in_dim=131, hidden=128):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, 1)   # predict gain in dB
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            # Calculate feature dimension
            N_MELS = 64  # Same as in train.py
            feature_dim = 2*N_MELS + 3  # mean + std + [rms, crest, flatness]
            
            # Try to load the model
            model_path = os.path.join(audionorm_dl_path, "models", "norm_mlp.pth")
            if os.path.exists(model_path):
                self.model = MLP(in_dim=feature_dim)
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                self.N_MELS = N_MELS  # Store for feature extraction
                print("✅ Pre-trained model loaded successfully from:", model_path)
            else:
                print("⚠️  Pre-trained model not found at:", model_path, "Using fallback normalization.")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
    
    def extract_features(self, audio, sr):
        """Extract audio features for the model - matches training features in AudioNorm_DL/train.py"""
        try:
            # Extract log-mel spectrogram features
            n_mels = getattr(self, 'N_MELS', 64)  # Default to 64 if not set
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, power=2.0)
            S_db = librosa.power_to_db(S + 1e-12)
            mean = S_db.mean(axis=1)
            std = S_db.std(axis=1)
            
            # Simple loudness-related features
            rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
            peak = float(np.max(np.abs(audio)) + 1e-12)
            crest = float(peak / (rms + 1e-12))
            
            # Spectral flatness
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
            
            # Concatenate all features
            feat = np.concatenate([mean, std, np.array([rms, crest, flatness], dtype=np.float32)])
            return feat.astype(np.float32)
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            # Return zeros with the correct feature dimension
            n_mels = getattr(self, 'N_MELS', 64)
            return np.zeros(2*n_mels + 3, dtype=np.float32)

    def measure_lufs(self, audio, sr):
        """Measure LUFS using pyloudnorm"""
        try:
            meter = pyln.Meter(sr)
            return meter.integrated_loudness(audio)
        except Exception as e:
            print(f"LUFS measurement failed: {e}")
            return -23.0  # Default target

    def convert_to_wav_ffmpeg(self, input_path, output_path):
        """Convert audio to WAV using FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', 
                '-ar', '44100', '-ac', '2', '-y', output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"FFmpeg conversion failed: {e}")
            return False

    def normalize_audio_dl(self, audio, sr, target_lufs=-23.0):
        """Normalize audio using the DL model - matches the approach in AudioNorm_DL/serve.py"""
        if self.model is None or not AUDIO_DEPS_AVAILABLE:
            # Fallback to basic normalization
            return self.normalize_audio_basic(audio, sr, target_lufs)
        
        try:
            # Extract features matching the training format
            features = self.extract_features(audio, sr)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Predict gain using the model
            with torch.no_grad():
                predicted_gain = self.model(features_tensor).item()
            
            # Measure original LUFS
            meter = pyln.Meter(sr)
            original_lufs = meter.integrated_loudness(audio)
            
            if original_lufs == float('-inf'):
                print("Warning: Audio is silent")
                return audio
            
            # Calculate ideal gain (perfect target)
            ideal_gain = target_lufs - original_lufs
            
            # Blend model prediction with ideal (like in serve.py)
            blended_gain = 0.7 * predicted_gain + 0.3 * ideal_gain
            
            # Apply gain
            normalized_audio = audio * (10 ** (blended_gain / 20))
            
            # Optional: Exact refinement (measure again and apply residual gain)
            refined_lufs = meter.integrated_loudness(normalized_audio)
            if refined_lufs != float('-inf'):
                residual_gain = target_lufs - refined_lufs
                if abs(residual_gain) > 0.1:  # Only refine if difference is significant
                    normalized_audio = normalized_audio * (10 ** (residual_gain / 20))
            
            # Ensure no clipping
            max_val = np.max(np.abs(normalized_audio))
            if max_val > 1.0:
                normalized_audio = normalized_audio / max_val * 0.95
                
            return normalized_audio
        except Exception as e:
            print(f"DL normalization failed: {e}")
            return self.normalize_audio_basic(audio, sr, target_lufs)

    def normalize_audio_basic(self, audio, sr, target_lufs=-23.0):
        """Basic audio normalization using pyloudnorm"""
        try:
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            
            if loudness == float('-inf'):
                print("Warning: Audio is silent")
                return audio
                
            normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
            return normalized_audio
        except Exception as e:
            print(f"Basic normalization failed: {e}")
            return audio

    async def analyze_audio_file(self, file: UploadFile) -> AudioAnalysisResult:
        """Analyze audio file properties"""
        if not AUDIO_DEPS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Audio processing dependencies not available")
        
        with tempfile.NamedTemporaryFile(suffix=f".{file.filename.split('.')[-1]}", delete=False) as temp_input:
            temp_input.write(await file.read())
            temp_input_path = temp_input.name
        
        temp_wav_path = None
        
        try:
            # Handle different audio formats
            file_ext = file.filename.split('.')[-1].lower()
            
            if file_ext == 'wav':
                process_path = temp_input_path
            else:
                temp_wav_path = tempfile.mktemp(suffix=".wav")
                if not self.convert_to_wav_ffmpeg(temp_input_path, temp_wav_path):
                    raise HTTPException(status_code=400, detail=f"Failed to convert {file_ext} to WAV")
                process_path = temp_wav_path
            
            # Load and analyze audio
            audio, sr = librosa.load(process_path, sr=None, mono=False)
            
            # Convert to mono for analysis
            if audio.ndim > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio
            
            # Calculate properties
            duration = len(audio_mono) / sr
            lufs = self.measure_lufs(audio_mono, sr)
            rms = np.sqrt(np.mean(audio_mono**2))
            peak = np.max(np.abs(audio_mono))
            
            return AudioAnalysisResult(
                filename=file.filename,
                duration_seconds=round(duration, 2),
                sample_rate=sr,
                channels=audio.ndim if audio.ndim <= 2 else 2,
                lufs=round(lufs, 2) if lufs != float('-inf') else None,
                rms=round(float(rms), 4),
                peak=round(float(peak), 4),
                peak_db=round(20 * np.log10(peak) if peak > 0 else -float('inf'), 2),
                format=file_ext.upper()
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        finally:
            # Cleanup
            try:
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
            except:
                pass

    async def normalize_audio_file(
        self, 
        file: UploadFile, 
        target_lufs: float = -23.0, 
        use_dl_model: bool = True,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> tuple[str, AudioNormalizationResult]:
        """
        Normalize audio file and store result in database
        Returns: (output_file_path, normalization_result)
        """
        print(f"SERVICE: Starting normalization - target: {target_lufs} LUFS, file: {file.filename}")
        
        if not AUDIO_DEPS_AVAILABLE:
            print("SERVICE ERROR: Audio processing dependencies not available")
            raise HTTPException(status_code=503, detail="Audio processing dependencies not available")
        
        start_time = time.time()
        
        # Create temporary files
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{file.filename.split('.')[-1]}", delete=False) as temp_input:
                content = await file.read()
                if not content:
                    print("SERVICE ERROR: Uploaded file is empty")
                    raise HTTPException(status_code=400, detail="Uploaded file is empty")
                
                print(f"SERVICE: Read {len(content)} bytes from uploaded file")
                temp_input.write(content)
                temp_input_path = temp_input.name
                
            print(f"SERVICE: Created temp file at {temp_input_path}")
        except Exception as e:
            print(f"SERVICE ERROR: Failed to create temp file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")
        
        temp_wav_path = None
        temp_output_path = tempfile.mktemp(suffix=".wav")
        
        try:
            # Handle different audio formats
            file_ext = file.filename.split('.')[-1].lower()
            
            if file_ext == 'wav':
                # Direct WAV processing
                process_path = temp_input_path
            else:
                # Convert to WAV first
                temp_wav_path = tempfile.mktemp(suffix=".wav")
                if not self.convert_to_wav_ffmpeg(temp_input_path, temp_wav_path):
                    raise HTTPException(status_code=400, detail=f"Failed to convert {file_ext} to WAV")
                process_path = temp_wav_path
            
            # Load and analyze original audio
            audio, sr = librosa.load(process_path, sr=None, mono=False)
            
            # Ensure mono for processing
            if audio.ndim > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio
            
            # Analyze original audio
            original_lufs = self.measure_lufs(audio_mono, sr)
            original_rms = np.sqrt(np.mean(audio_mono**2))
            original_peak = np.max(np.abs(audio_mono))
            
            # Choose normalization method
            if use_dl_model and self.model is not None:
                normalized_audio = self.normalize_audio_dl(audio_mono, sr, target_lufs)
                method = "DL Model"
                used_dl = True
            else:
                normalized_audio = self.normalize_audio_basic(audio_mono, sr, target_lufs)
                method = "Basic"
                used_dl = False
            
            # Analyze normalized audio
            final_lufs = self.measure_lufs(normalized_audio, sr)
            final_rms = np.sqrt(np.mean(normalized_audio**2))
            final_peak = np.max(np.abs(normalized_audio))
            
            # Save normalized audio to temp file
            sf.write(temp_output_path, normalized_audio, sr)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate unique file ID for storage
            file_id = str(uuid.uuid4())
            normalized_filename = f"normalized_{file_id}_{file.filename.split('.')[0]}.wav"
            storage_file_path = os.path.join(STORAGE_PATH, normalized_filename)
            
            # Copy the normalized file to the permanent storage
            shutil.copy2(temp_output_path, storage_file_path)
            
            # Create normalization result
            result = AudioNormalizationResult(
                user_id=user_id,
                original_filename=file.filename,
                normalized_filename=normalized_filename,
                file_format=file_ext.upper(),
                file_size_bytes=len(content),
                duration_seconds=round(len(audio_mono) / sr, 2),
                sample_rate=sr,
                channels=audio.ndim if audio.ndim <= 2 else 2,
                original_lufs=round(original_lufs, 2) if original_lufs != float('-inf') else None,
                target_lufs=target_lufs,
                final_lufs=round(final_lufs, 2) if final_lufs != float('-inf') else None,
                original_peak=round(float(original_peak), 4),
                final_peak=round(float(final_peak), 4),
                rms_original=round(float(original_rms), 4),
                rms_final=round(float(final_rms), 4),
                normalization_method=method,
                processing_time_seconds=round(processing_time, 2),
                used_dl_model=used_dl,
                ip_address=ip_address,
                user_agent=user_agent,
                # Storage information
                storage_path=storage_file_path,
                file_id=file_id,
                is_stored=True
            )
            
            # Store in database
            collection = await self._get_collection()
            insert_result = await collection.insert_one(result.dict(by_alias=True))
            result.id = insert_result.inserted_id
            
            return temp_output_path, result
            
        except Exception as e:
            # Cleanup on error
            try:
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Normalization failed: {str(e)}")

    async def get_normalization_history(self, user_id: Optional[str] = None, limit: int = 50) -> List[AudioNormalizationResult]:
        """Get normalization history for user or all (if admin)"""
        collection = await self._get_collection()
        
        query = {}
        if user_id:
            query["user_id"] = user_id
        
        cursor = collection.find(query).sort("created_at", -1).limit(limit)
        results = []
        
        async for doc in cursor:
            results.append(AudioNormalizationResult(**doc))
        
        return results

    async def get_normalization_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get normalization statistics"""
        collection = await self._get_collection()
        
        query = {}
        if user_id:
            query["user_id"] = user_id
        
        # Basic stats
        total_count = await collection.count_documents(query)
        
        # Aggregation pipeline for more detailed stats
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": None,
                    "total_processing_time": {"$sum": "$processing_time_seconds"},
                    "avg_processing_time": {"$avg": "$processing_time_seconds"},
                    "total_duration": {"$sum": "$duration_seconds"},
                    "dl_model_usage": {"$sum": {"$cond": ["$used_dl_model", 1, 0]}},
                    "formats": {"$push": "$file_format"}
                }
            }
        ]
        
        result = await collection.aggregate(pipeline).to_list(1)
        stats = result[0] if result else {}
        
        return {
            "total_normalizations": total_count,
            "total_processing_time_seconds": round(stats.get("total_processing_time", 0), 2),
            "average_processing_time_seconds": round(stats.get("avg_processing_time", 0), 2),
            "total_audio_duration_seconds": round(stats.get("total_duration", 0), 2),
            "dl_model_usage_count": stats.get("dl_model_usage", 0),
            "basic_normalization_count": total_count - stats.get("dl_model_usage", 0),
            "most_common_formats": list(set(stats.get("formats", [])))
        }
    
    async def get_stored_audio_file(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a stored audio file by its result ID
        Returns file information including path and metadata
        """
        from bson import ObjectId
        
        try:
            # Get the normalization result from the database
            collection = await self._get_collection()
            result = await collection.find_one({"_id": ObjectId(result_id)})
            
            if not result:
                return None
                
            # Check if the file exists in storage
            storage_path = result.get("storage_path")
            if not storage_path or not os.path.exists(storage_path):
                # File not found in storage
                return {
                    "result": result,
                    "file_exists": False,
                    "error": "File not found in storage"
                }
                
            return {
                "result": result,
                "file_exists": True,
                "file_path": storage_path,
                "filename": result["normalized_filename"],
                "file_size_bytes": os.path.getsize(storage_path)
            }
                
        except Exception as e:
            print(f"Error retrieving stored file: {e}")
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get audio processing system status"""
        return {
            "status": "running",
            "audio_deps_available": AUDIO_DEPS_AVAILABLE,
            "model_loaded": self.model is not None,
            "supported_formats": ["wav", "mp3", "flac", "m4a"] if AUDIO_DEPS_AVAILABLE else ["wav"],
            "normalization_methods": ["DL Model", "Basic"] if self.model else ["Basic"],
            "endpoints": [
                "/audio/status - Get system status",
                "/audio/normalize/{target_lufs} - Normalize audio file",
                "/audio/normalize/10 - Normalize to -10 LUFS",
                "/audio/normalize/12 - Normalize to -12 LUFS", 
                "/audio/normalize/14 - Normalize to -14 LUFS",
                "/audio/analyze - Analyze audio properties",
                "/audio/history - Get normalization history",
                "/audio/stats - Get usage statistics"
            ]
        }
        
    # Special target LUFS methods with higher precision for music mastering
    async def normalize_to_10lufs(self, file: UploadFile, user_id: Optional[str] = None, 
                                 ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Normalize audio to -10 LUFS (louder masters, streaming platforms)"""
        return await self.normalize_audio_file(
            file, target_lufs=-10.0, use_dl_model=True,
            user_id=user_id, ip_address=ip_address, user_agent=user_agent
        )
    
    async def normalize_to_12lufs(self, file: UploadFile, user_id: Optional[str] = None, 
                                 ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Normalize audio to -12 LUFS (moderate loudness for streaming)"""
        return await self.normalize_audio_file(
            file, target_lufs=-12.0, use_dl_model=True,
            user_id=user_id, ip_address=ip_address, user_agent=user_agent
        )
    
    async def normalize_to_14lufs(self, file: UploadFile, user_id: Optional[str] = None, 
                                 ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Normalize audio to -14 LUFS (standard for many streaming platforms)"""
        return await self.normalize_audio_file(
            file, target_lufs=-14.0, use_dl_model=True,
            user_id=user_id, ip_address=ip_address, user_agent=user_agent
        )

# Create singleton instance
audio_service = AudioService()
print("✅ AudioService singleton initialized")

# Print some diagnostic info at import time
try:
    if AUDIO_DEPS_AVAILABLE:
        print("✅ Audio dependencies available:")
        print(f"   - librosa: {librosa.__version__}")
        print(f"   - pyloudnorm: {pyln.__version__}")
        print(f"   - torch: {torch.__version__}")
        print(f"   - soundfile: {sf.__version__}")
        print(f"   - numpy: {np.__version__}")
    else:
        print("⚠️ Audio processing dependencies missing!")
        try:
            import pkg_resources
            print("Available packages:")
            for pkg in pkg_resources.working_set:
                print(f"   - {pkg.key}: {pkg.version}")
        except:
            pass
except Exception as e:
    print(f"⚠️ Error printing audio dependencies: {e}")
audio_service = AudioService()
