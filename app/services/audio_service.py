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

# AudioNorm_DL path setup for model imports
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
audionorm_dl_path = os.path.join(current_dir, "AudioNorm_DL")
if audionorm_dl_path not in sys.path and os.path.exists(audionorm_dl_path):
    sys.path.append(audionorm_dl_path)

# Optional imports - will be handled gracefully if not available
try:
    import librosa
    import torch
    import torch.nn as nn
    import soundfile as sf
    # pyloudnorm is only used for LUFS measurement in the DL model
    import pyloudnorm as pyln
    AUDIO_DEPS_AVAILABLE = True
    print("✅ Audio processing dependencies loaded successfully")
except ImportError as e:
    AUDIO_DEPS_AVAILABLE = False
    print(f"⚠️ Audio processing dependencies not available: {e}")
    print("Missing dependencies. Please install: pip install torch librosa soundfile pyloudnorm")

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
            # CNN model matching the one from AudioNorm_DL/train.py
            class AudioNormCNN(nn.Module):
                def __init__(self, n_mels=64, additional_features_dim=9):
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
            
            # Model parameters
            N_MELS = 64  # Same as in train.py
            TARGET_TIME_FRAMES = 128  # Fixed time dimension for CNN
            
            # Try to load the model - check for both CNN and MLP model files
            cnn_model_path = os.path.join(audionorm_dl_path, "models", "norm_cnn.pth")
            # mlp_model_path = os.path.join(audionorm_dl_path, "models", "norm_mlp.pth")
            
            if os.path.exists(cnn_model_path):
                self.model = AudioNormCNN(n_mels=N_MELS, additional_features_dim=9)
                self.model.load_state_dict(torch.load(cnn_model_path, map_location='cpu'))
                self.model.eval()
                self.N_MELS = N_MELS  # Store for feature extraction
                self.TARGET_TIME_FRAMES = TARGET_TIME_FRAMES  # Store for feature extraction
                print("✅ Pre-trained CNN model loaded successfully from:", cnn_model_path)
            else:
                print("⚠️  No trained model found. Please train a model first.")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
    
    def extract_features(self, audio, sr):
        """Extract spectrogram features for the CNN model - matches training features in AudioNorm_DL/train.py"""
        try:
            # Extract mel-spectrogram for CNN processing
            n_mels = getattr(self, 'N_MELS', 64)
            target_time_frames = getattr(self, 'TARGET_TIME_FRAMES', 128)
            
            # Use shorter hop length for better time resolution (matching train.py)
            hop_length = 512
            n_fft = 2048
            
            S = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, 
                hop_length=hop_length, power=2.0
            )
            S_db = librosa.power_to_db(S + 1e-12)
            
            # Normalize the spectrogram
            S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
            
            # Resize to fixed time dimension
            if S_db.shape[1] < target_time_frames:
                # Pad with zeros if too short
                pad_width = target_time_frames - S_db.shape[1]
                S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
            elif S_db.shape[1] > target_time_frames:
                # Crop if too long (take center portion)
                start = (S_db.shape[1] - target_time_frames) // 2
                S_db = S_db[:, start:start + target_time_frames]
            
            # Return as (n_mels, time_frames) for CNN
            return S_db.astype(np.float32)
        except Exception as e:
            print(f"Spectrogram extraction failed: {e}")
            # Return zeros with the correct dimensions
            n_mels = getattr(self, 'N_MELS', 64)
            target_time_frames = getattr(self, 'TARGET_TIME_FRAMES', 128)
            return np.zeros((n_mels, target_time_frames), dtype=np.float32)

    def extract_additional_features(self, audio, sr):
        """Extract additional scalar features for concatenation - matches training features in AudioNorm_DL/train.py"""
        try:
            # Basic audio features
            rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
            peak = float(np.max(np.abs(audio)) + 1e-12)
            crest = float(peak / (rms + 1e-12))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Aggregate temporal features (9 features total)
            features = np.array([
                rms, peak, crest,
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
            ], dtype=np.float32)
            
            return features
        except Exception as e:
            print(f"Additional feature extraction failed: {e}")
            # Return zeros with the correct feature dimension (9 features)
            return np.zeros(9, dtype=np.float32)

    def measure_lufs(self, audio, sr):
        """Measure LUFS using pyloudnorm"""
        try:
            meter = pyln.Meter(sr)
            return meter.integrated_loudness(audio)
        except Exception as e:
            print(f"LUFS measurement failed: {e}")
            return -23.0  # Default target

    def convert_to_wav_ffmpeg(self, input_path, output_path):
        """Convert audio to WAV using FFmpeg (fallback method)"""
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

    def load_audio_file(self, file_path):
        """Load audio file using librosa with fallback support for different formats"""
        try:
            # Try to load directly with librosa (supports many formats including MP3)
            print(f"Attempting to load audio file: {file_path}")
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            print(f"Successfully loaded audio: {audio.shape if hasattr(audio, 'shape') else len(audio)} samples at {sr}Hz")
            return audio, sr, True
        except Exception as e:
            print(f"Failed to load audio with librosa: {e}")
            return None, None, False

    def normalize_audio_dl(self, audio, sr, target_lufs=-23.0):
        """Normalize audio using the CNN DL model - matches the approach in AudioNorm_DL/serve.py"""
        if self.model is None or not AUDIO_DEPS_AVAILABLE:
            raise HTTPException(status_code=503, detail="DL model not available for normalization")
        
        try:
            # Extract features matching the training format
            spectrogram = self.extract_features(audio, sr)  # (n_mels, time_frames)
            additional_features = self.extract_additional_features(audio, sr)  # (9,)
            
            # Convert to tensors and add batch dimension
            spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0)  # (1, n_mels, time_frames)
            additional_features_tensor = torch.FloatTensor(additional_features).unsqueeze(0)  # (1, 9)
            
            # Predict gain using the model
            with torch.no_grad():
                predicted_gain = self.model(spectrogram_tensor, additional_features_tensor).item()
            
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
            raise HTTPException(status_code=500, detail=f"DL normalization failed: {str(e)}")



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
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        original_file_id: Optional[str] = None
    ) -> tuple[str, AudioNormalizationResult]:
        """
        Optimized: Normalize audio file and store result in database with reduced file I/O and temp file usage.
        Returns: (output_file_path, normalization_result)
        """
        print(f"SERVICE: Starting normalization - target: {target_lufs} LUFS, file: {file.filename}")
        if not AUDIO_DEPS_AVAILABLE:
            print("SERVICE ERROR: Audio processing dependencies not available")
            raise HTTPException(status_code=503, detail="Audio processing dependencies not available")

        start_time = time.time()
        content = await file.read()
        if not content:
            print("SERVICE ERROR: Uploaded file is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        file_ext = file.filename.split('.')[-1].lower()
        temp_input_path = None
        temp_wav_path = None
        temp_output_path = None
        try:
            # Try to load audio directly from memory using soundfile (for wav/flac/ogg)
            audio = None
            sr = None
            load_success = False
            import io
            try:
                if file_ext in ["wav", "flac", "ogg"]:
                    with io.BytesIO(content) as buf:
                        audio_data, sr = sf.read(buf, always_2d=False)
                        # Convert to float32 if needed
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32)
                        audio = audio_data
                        load_success = True
            except Exception as e:
                print(f"SoundFile in-memory load failed: {e}")

            # If not loaded, fallback to librosa (in-memory)
            if not load_success:
                try:
                    with io.BytesIO(content) as buf:
                        audio, sr = librosa.load(buf, sr=None, mono=False)
                        load_success = True
                except Exception as e:
                    print(f"Librosa in-memory load failed: {e}")

            # If still not loaded, fallback to FFmpeg conversion (requires temp file)
            if not load_success:
                with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as temp_input:
                    temp_input.write(content)
                    temp_input_path = temp_input.name
                temp_wav_path = tempfile.mktemp(suffix=".wav")
                if self.convert_to_wav_ffmpeg(temp_input_path, temp_wav_path):
                    try:
                        audio, sr = sf.read(temp_wav_path, always_2d=False)
                        if audio.dtype != np.float32:
                            audio = audio.astype(np.float32)
                        load_success = True
                    except Exception as e:
                        print(f"SoundFile load after FFmpeg failed: {e}")
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to process {file_ext} file - FFmpeg not available or conversion failed")

            if not load_success or audio is None or sr is None:
                raise HTTPException(status_code=400, detail=f"Failed to load audio file: {file.filename}")

            print(f"Successfully loaded audio: {audio.shape if hasattr(audio, 'shape') else len(audio)} samples at {sr}Hz")

            # Ensure mono for processing
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio

            # Analyze original audio
            original_lufs = self.measure_lufs(audio_mono, sr)
            original_rms = np.sqrt(np.mean(audio_mono**2))
            original_peak = np.max(np.abs(audio_mono))

            # Use CNN DL model for normalization
            if self.model is not None:
                normalized_audio = self.normalize_audio_dl(audio_mono, sr, target_lufs)
                method = "CNN DL Model"
                used_dl = True
            else:
                raise HTTPException(status_code=503, detail="CNN DL model not available - please ensure norm_cnn.pth exists")

            # Analyze normalized audio
            final_lufs = self.measure_lufs(normalized_audio, sr)
            final_rms = np.sqrt(np.mean(normalized_audio**2))
            final_peak = np.max(np.abs(normalized_audio))

            # Save normalized audio to temp file for GridFS upload
            temp_output_path = tempfile.mktemp(suffix=".wav")
            sf.write(temp_output_path, normalized_audio, sr)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Generate unique file ID for storage
            file_id = str(uuid.uuid4())
            normalized_filename = f"normalized_{file_id}_{file.filename.split('.')[0]}.wav"

            # Store normalized file in GridFS
            from motor.motor_asyncio import AsyncIOMotorGridFSBucket
            db = await get_db()
            bucket = AsyncIOMotorGridFSBucket(db)
            with open(temp_output_path, 'rb') as f:
                normalized_audio_data = f.read()
            gridfs_id = await bucket.upload_from_stream(
                normalized_filename,
                normalized_audio_data,
                metadata={
                    "user_id": user_id,
                    "original_filename": file.filename,
                    "target_lufs": target_lufs,
                    "final_lufs": round(final_lufs, 2) if final_lufs != float('-inf') else None,
                    "processing_method": method,
                    "created_at": datetime.utcnow()
                }
            )

            # Create normalization result
            result = AudioNormalizationResult(
                user_id=user_id,
                original_filename=file.filename,
                normalized_filename=normalized_filename,
                file_format=file_ext.upper(),
                file_size_bytes=len(content),
                duration_seconds=round(len(audio_mono) / sr, 2),
                sample_rate=sr,
                channels=audio.shape[0] if isinstance(audio, np.ndarray) and audio.ndim == 2 else 1,
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
                storage_path=None,
                file_id=str(gridfs_id),
                is_stored=True
            )

            # Store in database
            collection = await self._get_collection()
            result_dict = result.model_dump(by_alias=True, exclude={'id'})
            result_dict["gridfs_id"] = gridfs_id
            if original_file_id:
                result_dict["original_file_id"] = original_file_id
            insert_result = await collection.insert_one(result_dict)
            result.id = insert_result.inserted_id

            return temp_output_path, result

        except Exception as e:
            print(f"SERVICE ERROR: {e}")
            raise HTTPException(status_code=500, detail=f"Normalization failed: {str(e)}")
        finally:
            # Always clean up temp files
            for path in [temp_input_path, temp_wav_path, temp_output_path]:
                try:
                    if path and os.path.exists(path):
                        os.unlink(path)
                except Exception as cleanup_error:
                    print(f"Warning: Cleanup failed: {cleanup_error}")

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
            "most_common_formats": list(set(stats.get("formats", []))),
            "normalization_method": "CNN DL Model"
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
            "audio_dependencies_available": AUDIO_DEPS_AVAILABLE,  # Fixed key name
            "model_loaded": self.model is not None,
            "supported_formats": ["wav", "mp3", "flac", "m4a"] if AUDIO_DEPS_AVAILABLE else [],
            "normalization_methods": ["CNN DL Model"] if self.model else ["None - CNN Model Required"],
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
            file, target_lufs=-10.0,
            user_id=user_id, ip_address=ip_address, user_agent=user_agent
        )
    
    async def normalize_to_12lufs(self, file: UploadFile, user_id: Optional[str] = None, 
                                 ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Normalize audio to -12 LUFS (moderate loudness for streaming)"""
        return await self.normalize_audio_file(
            file, target_lufs=-12.0,
            user_id=user_id, ip_address=ip_address, user_agent=user_agent
        )
    
    async def normalize_to_14lufs(self, file: UploadFile, user_id: Optional[str] = None, 
                                 ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Normalize audio to -14 LUFS (standard for many streaming platforms)"""
        return await self.normalize_audio_file(
            file, target_lufs=-14.0,
            user_id=user_id, ip_address=ip_address, user_agent=user_agent
        )

# Create singleton instance
audio_service = AudioService()
print("✅ AudioService singleton initialized")

# Print some diagnostic info at import time
try:
    if AUDIO_DEPS_AVAILABLE:
        print("✅ Audio dependencies available:")
        try:
            print(f"   - librosa: {librosa.__version__}")
        except:
            print("   - librosa: version unknown")
        try:
            print(f"   - pyloudnorm: {pyln.__version__}")
        except:
            print("   - pyloudnorm: version unknown")
        try:
            print(f"   - torch: {torch.__version__}")
        except:
            print("   - torch: version unknown")
        try:
            print(f"   - soundfile: {sf.__version__}")
        except:
            print("   - soundfile: version unknown")
        try:
            print(f"   - numpy: {np.__version__}")
        except:
            print("   - numpy: version unknown")
    else:
        print("⚠️ Audio processing dependencies missing!")
        print("Required packages: librosa, pyloudnorm, torch, soundfile, numpy")
except Exception as e:
    print(f"⚠️ Error printing audio dependencies: {e}")
audio_service = AudioService()
