"""
Audio Normalization Model
Defines the database schema for audio normalization results
"""
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from bson import ObjectId

"""
Audio Normalization Model
Defines the database schema for audio normalization results
"""
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from bson import ObjectId

class AudioNormalizationResult(BaseModel):
    """Model for audio normalization results stored in database"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    user_id: Optional[str] = None  # Optional for anonymous usage
    original_filename: str
    normalized_filename: str
    file_format: str
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    
    # Audio analysis data
    original_lufs: Optional[float] = None
    target_lufs: float
    final_lufs: Optional[float] = None
    original_peak: float
    final_peak: float
    rms_original: float
    rms_final: float
    
    # Processing info
    normalization_method: str  # "DL Model" or "Basic"
    processing_time_seconds: float
    used_dl_model: bool
    
    # File storage information
    storage_path: Optional[str] = None
    file_id: Optional[str] = None
    is_stored: bool = False
    
    # Reference to original uploaded file (for on-demand regeneration)
    original_upload_id: Optional[str] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class AudioAnalysisResult(BaseModel):
    """Model for audio analysis without normalization"""
    filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    lufs: Optional[float]
    rms: float
    peak: float
    peak_db: float
    format: str

class NormalizationRequest(BaseModel):
    """Request model for normalization"""
    target_lufs: float = Field(default=-23.0, ge=-70, le=0, description="Target LUFS level")
    use_dl_model: bool = Field(default=True, description="Use DL model if available")
    
class NormalizationResponse(BaseModel):
    """Response model for normalization"""
    success: bool
    message: str
    result_id: Optional[str] = None
    filename: str
    processing_time: float
    audio_info: AudioAnalysisResult
