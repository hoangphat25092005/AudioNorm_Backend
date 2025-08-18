"""
Audio Normalization Controller
Handles all audio normalization operations
"""
from fastapi import APIRouter, UploadFile, File, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId
from datetime import datetime
import tempfile
import os

from app.config.database import get_db
from app.config.jwt_dependency import get_current_user
from app.services.audio_service import audio_service

router = APIRouter(tags=["Normalization"])


@router.post("/normalize-uploaded/{file_id}/{target_lufs}")
async def normalize_uploaded_file(file_id: str, target_lufs: float = -23.0, request: Request = None, current_user_id: str = Depends(get_current_user)):
    """
    Normalize a previously uploaded file from GridFS by its file ID
    
    Parameters:
    - file_id: The ID of the uploaded file in the audio_files collection
    - target_lufs: Target loudness level in LUFS (default: -23.0 LUFS)
    
    Returns information about the normalization process. File is stored in database.
    """
    
    try:
        print(f"API: Normalize uploaded file request - file_id: {file_id}, target: {target_lufs} LUFS")
        
        # Validate target LUFS range
        if target_lufs > 0 or target_lufs < -70:
            return JSONResponse(
                status_code=400,
                content={"error": "target_lufs must be between -70 and 0 (e.g., -14, -16, -23)"}
            )

        # Get the uploaded file from database
        db = await get_db()
        bucket = AsyncIOMotorGridFSBucket(db)
        
        # Find the file metadata
        file_doc = await db["audio_files"].find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            return JSONResponse(
                status_code=404,
                content={"error": f"File with ID {file_id} not found"}
            )
        
        # Check if the file belongs to the current user
        if file_doc.get("user_id") != current_user_id:
            return JSONResponse(
                status_code=403,
                content={"error": "You can only normalize your own files"}
            )
        
        gridfs_id = file_doc.get("gridfs_id")
        if not gridfs_id:
            return JSONResponse(
                status_code=404,
                content={"error": "File data not found in storage"}
            )
        
        # Stream the file from GridFS directly to a temp file to avoid loading into memory
        try:
            import io
            original_filename = file_doc.get("filename", "unknown.mp3")
            file_extension = original_filename.split(".")[-1].lower() if "." in original_filename else "mp3"
            import os
            fd, temp_file_path = tempfile.mkstemp(suffix=f".{file_extension}")
            os.close(fd)
            # Download directly to file
            with open(temp_file_path, "wb") as f:
                await bucket.download_to_stream(gridfs_id, f)
        except Exception as e:
            return JSONResponse(
                status_code=404,
                content={"error": f"Could not retrieve file data: {str(e)}"}
            )
        # Check if file is empty
        if os.path.getsize(temp_file_path) == 0:
            os.unlink(temp_file_path)
            return JSONResponse(
                status_code=400,
                content={"error": "File content is empty"}
            )
        # Get client info
        client_host = request.client.host if request and request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown") if request else "unknown"
        
        # Use audio service for normalization
        try:
            service_status = audio_service.get_system_status()
            if service_status.get("audio_dependencies_available", False):
                service_to_use = audio_service
                service_type = "full"
                print("Using full audio service for analysis")
            else:
                raise Exception("Audio dependencies not available in full service")
        except Exception as e:
            print(f"Audio service not available: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Audio normalization service is not available", 
                    "details": "Audio processing dependencies are not installed or not working",
                    "suggestion": "Please ensure librosa, pyloudnorm, and other audio dependencies are properly installed"
                }
            )
        
        # Use the service to analyze the file
        try:
            # Create a file-like object for the temp file
            class MockUploadFile:
                def __init__(self, filename, file_path):
                    self.filename = filename
                    self.file_path = file_path
                    self.file = open(file_path, "rb")
                async def read(self):
                    self.file.seek(0)
                    return self.file.read()
                def seek(self, position):
                    self.file.seek(position)
                def close(self):
                    self.file.close()
            mock_file = MockUploadFile(original_filename, temp_file_path)
            # Process and store in database
            temp_output_path, result = await service_to_use.normalize_audio_file(
                file=mock_file,
                target_lufs=target_lufs,
                user_id=current_user_id,
                ip_address=client_host,
                user_agent=user_agent,
                original_file_id=str(file_id)
            )
            mock_file.close()
            
            # Add reference to original uploaded file
            result.original_upload_id = str(file_id)
            result.is_stored = True  # Mark as stored in database
            result.storage_path = None  # No file system storage
            
            # Update the original file record with normalization metadata
            normalization_data = {
                "status": "normalized",
                "target_lufs": result.target_lufs,
                "final_lufs": result.final_lufs,
                "original_lufs": result.original_lufs,
                "normalized_filename": result.normalized_filename,
                "normalization_method": result.normalization_method,
                "processing_time_seconds": result.processing_time_seconds,
                "normalized_at": datetime.utcnow(),
                "ready_to_download": True
            }
            
            # Update the original file record with normalization metadata
            await db["audio_files"].update_one(
                {"_id": ObjectId(file_id)},
                {"$set": normalization_data}
            )
            
            print(f"Updated original file {file_id} with normalization metadata")
            print(f"Normalization completed with ID: {result.id}")
            
            # Get the base URL for download links
            base_url = str(request.base_url).rstrip('/') if request else "http://localhost:8000"
            
            # Clean up temporary files
            for temp_path in [temp_file_path, temp_output_path]:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"Temporary file removed: {temp_path}")
            
            # Return detailed information about the normalization
            response_data = {
                "status": "success",
                "message": f"Audio normalized and stored in database at {target_lufs} LUFS",
                "service_type": "full",
                "result_id": str(result.id),  # Use the normalized file ID for export
                "original_filename": result.original_filename,
                "normalized_filename": result.normalized_filename,
                "original_lufs": result.original_lufs,
                "target_lufs": result.target_lufs,
                "final_lufs": result.final_lufs,
                "normalization_method": result.normalization_method,
                "processing_time_seconds": result.processing_time_seconds,
                "download_url": f"{base_url}/audio/export/{result.id}",  # Use normalized file ID for export
                "ready_to_download": True
            }
            
            return response_data
            
        except Exception as process_error:
            print(f"Error in audio processing: {str(process_error)}")
            import traceback
            traceback.print_exc()
            
            # Clean up temp file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise
        
    except Exception as e:
        print(f"Unhandled exception in normalize_uploaded_file: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ensure temp file cleanup on error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": f"An error occurred during audio normalization: {str(e)}",
                "traceback": traceback.format_exc()
            }
        )

@router.post("/normalize/{target_lufs}")
async def normalize_audio(request: Request, target_lufs: float = -23.0, file: UploadFile = File(...), current_user_id: str = Depends(get_current_user)):
    """
    Upload and normalize an audio file to the specified LUFS target
    
    Parameters:
    - target_lufs: Target loudness level in LUFS (default: -23.0 LUFS)
    - file: The audio file to normalize
    
    Returns information about the normalization process and download link.
    """
    
    # Validate target LUFS range
    if target_lufs > 0 or target_lufs < -70:
        return JSONResponse(
            status_code=400,
            content={"error": "target_lufs must be between -70 and 0 (e.g., -14, -16, -23)"}
        )
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg')):
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported file format. Please upload MP3, WAV, FLAC, M4A, AAC, or OGG files."}
        )
    
    try:
        # Get client info
        client_host = request.client.host if request and request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown") if request else "unknown"
        
        # Get audio service
        try:
            service_status = audio_service.get_system_status()
            if service_status.get("audio_dependencies_available", False):
                service_to_use = audio_service
                print("Using full audio service for normalization")
            else:
                raise Exception("Audio dependencies not available in full service")
        except Exception as e:
            print(f"Audio service not available: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Audio normalization service is not available", 
                    "details": "Audio processing dependencies are not installed or not working"
                }
            )
        
        # Process the file with the service
        try:
            temp_output_path, result = await service_to_use.normalize_audio_file(
                file=file,
                target_lufs=target_lufs,
                user_id=current_user_id,
                ip_address=client_host,
                user_agent=user_agent
            )
            
            print(f"File processed and stored in database")
            
            # Get the base URL for download links
            base_url = str(request.base_url).rstrip('/') if request else "http://localhost:8000"
            
            # Clean up temporary output file
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
                print(f"Temporary file removed: {temp_output_path}")
            
            # Return detailed information about the normalization
            response_data = {
                "status": "success",
                "message": f"Audio normalized and stored in database at {target_lufs} LUFS",
                "result_id": str(result.id),
                "original_filename": result.original_filename,
                "normalized_filename": result.normalized_filename,
                "original_lufs": result.original_lufs,
                "target_lufs": result.target_lufs,
                "final_lufs": result.final_lufs,
                "normalization_method": result.normalization_method,
                "processing_time_seconds": result.processing_time_seconds,
                "download_url": f"{base_url}/audio/export/{result.id}",
                "ready_to_download": True
            }
            
            return response_data
            
        except HTTPException as he:
            # Re-raise HTTP exceptions as-is
            raise he
        except Exception as process_error:
            error_detail = str(process_error)
            error_type = "Processing Error"
            
            print(f"Error in audio processing: {error_detail}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"{error_type}: {error_detail}",
                    "suggestion": "Please check your file format and try again"
                }
            )
        
    except Exception as e:
        print(f"Unhandled exception in normalize_audio: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during audio normalization: {str(e)}"}
        )

# Convenience endpoints for common LUFS values
@router.post("/normalize-10lufs")
async def normalize_10lufs(request: Request, file: UploadFile = File(...)):
    return await normalize_audio(request, -10.0, file)

@router.post("/normalize-12lufs")
async def normalize_12lufs(request: Request, file: UploadFile = File(...)):
    return await normalize_audio(request, -12.0, file)

@router.post("/normalize-14lufs")
async def normalize_14lufs(request: Request, file: UploadFile = File(...)):
    return await normalize_audio(request, -14.0, file)

@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file properties without normalization
    """
    try:
        # Use audio service for analysis
        analysis_result = await audio_service.analyze_audio_file(file)
        return {
            "status": "success",
            "analysis": analysis_result
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )
