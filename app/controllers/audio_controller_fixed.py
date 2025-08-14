from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import tempfile
import os
import shutil
from datetime import datetime
from typing import List
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId
from app.config.database import get_db

router = APIRouter(tags=["Audio Normalization"])

@router.get("/status")
async def get_audio_status():
    return {"status": "working", "message": "Audio API is operational"}

@router.get("/test")
async def test_audio():
    return {"test": "success", "api_version": "1.0.0"}

@router.get("/uploads")
async def list_recent_uploads(limit: int = 10):
    """List recent uploaded audio metadata from 'audio_files' collection."""
    try:
        db = await get_db()
        cursor = db["audio_files"].find({}).sort("created_at", -1).limit(limit)
        items = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])  # make JSON-friendly
            if "gridfs_id" in doc:
                doc["gridfs_id"] = str(doc["gridfs_id"]) 
            items.append(doc)
        return {"count": len(items), "items": items}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/dependencies")
async def test_audio_dependencies():
    """Test endpoint to check if audio processing dependencies are working"""
    from app.services.audio_service import audio_service
    
    # Check dependencies
    result = {
        "test": "running",
        "api_version": "1.0.0",
        "audio_dependencies": {}
    }
    
    try:
        import librosa
        result["audio_dependencies"]["librosa"] = {"available": True, "version": librosa.__version__}
    except ImportError as e:
        result["audio_dependencies"]["librosa"] = {"available": False, "error": str(e)}
    
    try:
        import pyloudnorm as pyln
        result["audio_dependencies"]["pyloudnorm"] = {"available": True, "version": pyln.__version__}
    except ImportError as e:
        result["audio_dependencies"]["pyloudnorm"] = {"available": False, "error": str(e)}
    
    try:
        import soundfile as sf
        result["audio_dependencies"]["soundfile"] = {"available": True, "version": sf.__version__}
    except ImportError as e:
        result["audio_dependencies"]["soundfile"] = {"available": False, "error": str(e)}
    
    try:
        import torch
        result["audio_dependencies"]["torch"] = {"available": True, "version": torch.__version__}
    except ImportError as e:
        result["audio_dependencies"]["torch"] = {"available": False, "error": str(e)}
    
    # Check service status
    try:
        result["service_status"] = audio_service.get_system_status()
    except Exception as e:
        result["service_status"] = {"error": str(e)}
    
    return result

@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and persist it to MongoDB (GridFS + metadata).
    Returns stored IDs so you can verify in the database.
    """
    try:
        # Validate file type
        allowed_extensions = ["wav", "mp3", "ogg", "flac", "m4a"]
        file_extension = (file.filename.split(".")[-1] if "." in file.filename else "").lower()
        if file_extension not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={"error": f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"}
            )

        # Read content
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty"})

        # Get DB and GridFS bucket
        db = await get_db()
        bucket = AsyncIOMotorGridFSBucket(db)

        # Store file in GridFS
        gridfs_id = await bucket.upload_from_stream(
            file.filename,
            content,
            metadata={
                "content_type": file.content_type or f"audio/{file_extension}",
                "extension": file_extension,
                "size": len(content),
            },
        )

        # Store metadata document
        meta_doc = {
            "filename": file.filename,
            "content_type": file.content_type or f"audio/{file_extension}",
            "size": len(content),
            "gridfs_id": gridfs_id,
            "status": "uploaded",
            "created_at": datetime.utcnow(),
        }
        meta_result = await db["audio_files"].insert_one(meta_doc)

        return {
            "status": "success",
            "message": "File uploaded and saved to database",
            "file": {
                "id": str(meta_result.inserted_id),
                "gridfs_id": str(gridfs_id),
                "filename": file.filename,
                "size": len(content),
                "content_type": file.content_type or f"audio/{file_extension}",
            },
            "note": "Normalization results are saved separately in 'audio_normalizations' when using /audio/normalize/{target_lufs}"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Upload failed: {str(e)}"}
        )

@router.get("/normalize/{target_lufs}")
async def normalize_audio_get(target_lufs: float = -23.0):
    """
    Test endpoint for normalization - GET method
    """
    return {
        "status": "error",
        "message": "This endpoint requires a POST request with an audio file",
        "target_lufs": target_lufs,
        "correct_usage": "POST /audio/normalize/{target_lufs} with a file upload",
        "example": "curl -X POST -F 'file=@your_audio.wav' http://localhost:8000/audio/normalize/-14"
    }

@router.post("/normalize/{target_lufs}")
async def normalize_audio(request: Request, target_lufs: float = -23.0, file: UploadFile = File(...)):
    """
    Upload and normalize an audio file to the specified LUFS target
    
    Parameters:
    - target_lufs: Target loudness level in LUFS (default: -23.0 LUFS)
    - file: The audio file to normalize
    
    Returns information about the normalization process and a download link.
    """
    from app.services.audio_service import audio_service
    
    try:
        print(f"API: Normalize request received - target: {target_lufs} LUFS, file: {file.filename}")
        
        # Validate target LUFS range (typical: -70..0)
        if target_lufs > 0 or target_lufs < -70:
            return JSONResponse(
                status_code=400,
                content={"error": "target_lufs must be between -70 and 0 (e.g., -14, -16, -23)"}
            )

        # Check if the file is an audio file
        allowed_extensions = ["wav", "mp3", "ogg", "flac", "m4a"]
        file_extension = file.filename.split(".")[-1].lower()
        
        if file_extension not in allowed_extensions:
            print(f"API: Unsupported file extension: {file_extension}")
            return JSONResponse(
                status_code=400,
                content={"error": f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"}
            )
        
        # Get client info for tracking
        client_host = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        print(f"Processing with client_host: {client_host}, user-agent: {user_agent}")
        
        # Use the audio service to normalize the file
        # In a complete implementation with authentication, you would pass the user_id
        try:
            temp_file_path, result = await audio_service.normalize_audio_file(
                file=file,
                target_lufs=target_lufs,
                use_dl_model=True,
                ip_address=client_host,
                user_agent=user_agent
            )
            
            print(f"Normalization successful. Result ID: {result.id}")
            print(f"File stored at: {result.storage_path}")
            print(f"Is stored: {result.is_stored}")
            
            # Get the base URL for download links
            base_url = str(request.base_url).rstrip('/')
            
            # Clean up the temporary file - we now have a stored version
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                print(f"Temporary file removed: {temp_file_path}")
            
            # Return detailed information about the normalization
            return {
                "status": "success",
                "message": f"Audio normalized to {target_lufs} LUFS successfully",
                "result_id": str(result.id),
                "original_filename": result.original_filename,
                "normalized_filename": result.normalized_filename,
                "original_lufs": result.original_lufs,
                "target_lufs": result.target_lufs,
                "final_lufs": result.final_lufs,
                "normalization_method": result.normalization_method,
                "processing_time_seconds": result.processing_time_seconds,
                "download_url": f"{base_url}/audio/download/{result.id}"
            }
        except Exception as process_error:
            print(f"Error in audio processing: {str(process_error)}")
            import traceback
            traceback.print_exc()
            raise
        
    except Exception as e:
        print(f"Unhandled exception in normalize_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ensure temp file cleanup on error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            
        return JSONResponse(
            status_code=500,
            content={
                "error": f"An error occurred during audio normalization: {str(e)}",
                "details": traceback.format_exc()
            }
        )

@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file properties without normalization
    
    This endpoint accepts audio files and returns analysis information such as:
    - Duration
    - Sample rate
    - File format
    - Loudness measurements (simulated)
    """
    try:
        # Check if the file is an audio file
        allowed_extensions = ["wav", "mp3", "ogg", "flac", "m4a"]
        file_extension = file.filename.split(".")[-1].lower()
        
        if file_extension not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={"error": f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"}
            )
        
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            # Read the uploaded file in chunks and write to the temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Get file info
        file_size = os.path.getsize(temp_file_path)
        
        # In a complete implementation:
        # 1. Use librosa to load and analyze the audio file
        # 2. Calculate LUFS, peak values, duration, etc.
        
        # For this testing API, we'll simulate the analysis
        analysis_result = {
            "filename": file.filename,
            "file_size_bytes": file_size,
            "file_type": file_extension,
            "analysis_time": datetime.now().isoformat(),
            "simulated_metrics": {
                "duration_seconds": 180.5,  # Simulated value
                "sample_rate": 44100,  # Simulated value
                "channels": 2,  # Simulated value
                "lufs": -18.5,  # Simulated value
                "peak_db": -3.2,  # Simulated value
                "dynamic_range_db": 15.7,  # Simulated value
            },
            "status": "success",
            "message": "Audio analysis completed successfully"
        }
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return analysis_result
        
    except Exception as e:
        # Ensure temp file cleanup on error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during analysis: {str(e)}"}
        )

# Add specialized endpoints for common LUFS targets
@router.post("/normalize/-10")
async def normalize_10lufs(request: Request, file: UploadFile = File(...)):
    """Normalize audio to -10 LUFS (louder masters, streaming platforms)"""
    return await normalize_audio(request=request, target_lufs=-10.0, file=file)

@router.post("/normalize/-12")
async def normalize_12lufs(request: Request, file: UploadFile = File(...)):
    """Normalize audio to -12 LUFS (moderate loudness for streaming)"""
    return await normalize_audio(request=request, target_lufs=-12.0, file=file)

@router.post("/normalize/-14")
async def normalize_14lufs(request: Request, file: UploadFile = File(...)):
    """Normalize audio to -14 LUFS (standard for many streaming platforms)"""
    return await normalize_audio(request=request, target_lufs=-14.0, file=file)

# Add an endpoint to list all normalized files for a user
@router.get("/history")
async def get_normalization_history(request: Request, limit: int = 50):
    """
    Get the normalization history for the current user
    
    Returns a list of previously normalized audio files
    """
    from app.services.audio_service import audio_service
    
    try:
        # In a complete implementation with authentication, you would get the user_id from the auth token
        # For now, we'll use the IP address as a simple identifier
        client_host = request.client.host if request.client else None
        
        # Get the normalization history from the service
        results = await audio_service.get_normalization_history(user_id=None, limit=limit)
        
        # Get the base URL for download links
        base_url = str(request.base_url).rstrip('/')
        
        # Format the results for the response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": str(result.id),
                "original_filename": result.original_filename,
                "normalized_filename": result.normalized_filename,
                "target_lufs": result.target_lufs,
                "original_lufs": result.original_lufs,
                "final_lufs": result.final_lufs,
                "duration_seconds": result.duration_seconds,
                "normalization_method": result.normalization_method,
                "processing_time_seconds": result.processing_time_seconds,
                "created_at": result.created_at.isoformat(),
                "download_url": f"{base_url}/audio/download/{result.id}"
            })
        
        return {
            "status": "success",
            "count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while retrieving history: {str(e)}"}
        )

# Add a download endpoint for the normalized audio files
@router.get("/download/{result_id}")
async def download_normalized_file(result_id: str):
    """
    Download a normalized audio file by its result ID
    
    This endpoint returns the normalized audio file for download.
    """
    from app.services.audio_service import audio_service
    
    try:
        # Get the file information from the service
        file_info = await audio_service.get_stored_audio_file(result_id)
        
        if not file_info:
            return JSONResponse(
                status_code=404,
                content={"error": f"Normalization result with ID {result_id} not found"}
            )
        
        if not file_info["file_exists"]:
            return JSONResponse(
                status_code=404,
                content={"error": "Audio file not found in storage"}
            )
        
        # Return the file as a downloadable response
        return FileResponse(
            path=file_info["file_path"],
            filename=file_info["filename"],
            media_type="audio/wav"
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while retrieving the file: {str(e)}"}
        )
