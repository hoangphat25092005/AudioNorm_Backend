
# A controller for the service that allows users to preview their normalized audio files
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.services.review_audio import stream_audio_preview, generate_normalized_preview, stream_original_audio_preview
from app.config.database import get_db
from app.config.jwt_dependency import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# Endpoint to preview the original (non-normalized) audio file
@router.get("/audio/preview/original/{file_id}", response_class=StreamingResponse)
async def preview_original_audio(
    file_id: str,
    db=Depends(get_db),
    user=Depends(get_current_user)
):
    """
    Preview the original (non-normalized) audio file for the owner.
    """
    logger.info(f"User {user} is previewing original audio file {file_id}")
    return await stream_original_audio_preview(db, file_id, user)

@router.get("/audio/preview/{file_id}", response_class=StreamingResponse)
async def preview_audio(
    file_id: str,
    db=Depends(get_db),
    user=Depends(get_current_user),
    target_lufs: str = None
):
    """
    Preview a normalized audio file if it exists, otherwise normalize the original file on-the-fly to the requested LUFS (if provided).
    """
    logger.info(f"User {user} is previewing audio file {file_id}")
    result = await stream_audio_preview(db, file_id, user)
    # If not found and target_lufs is provided, try on-the-fly normalization
    if hasattr(result, 'status_code') and getattr(result, 'status_code', 200) == 404 and target_lufs is not None:
        try:
            lufs_val = float(target_lufs)
        except Exception:
            return result  # fallback to 404 if not a valid float
        logger.info(f"Normalized file not found, generating preview on-the-fly for LUFS {lufs_val}")
        return await generate_normalized_preview(db, file_id, user, lufs_val)
    return result

