"""
Audio Upload Controller
Handles file uploads and listing uploaded files
"""
from fastapi import APIRouter, UploadFile, File, Depends, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId
from datetime import datetime
import os

from app.config.database import get_db
from app.config.jwt_dependency import get_current_user, get_current_user_flexible, get_current_user_flexible, get_current_user_flexible

router = APIRouter(tags=["Upload"])

@router.post("/upload")
async def upload_audio(file: UploadFile = File(...), current_user_id: str = Depends(get_current_user)):
    """
    Upload an audio file to GridFS for later processing
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg')):
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file format. Please upload MP3, WAV, FLAC, M4A, AAC, or OGG files."}
            )
        
        # Check file size (limit to 100MB)
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:  # 100MB
            return JSONResponse(
                status_code=400,
                content={"error": "File too large. Maximum size is 100MB."}
            )
        
        if len(content) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty file uploaded"}
            )
        
        # Store in GridFS
        db = await get_db()
        bucket = AsyncIOMotorGridFSBucket(db)
        
        # Upload to GridFS
        gridfs_id = await bucket.upload_from_stream(
            file.filename,
            content,
            metadata={
                "user_id": current_user_id,
                "content_type": file.content_type,
                "upload_time": datetime.utcnow()
            }
        )
        
        # Store metadata in audio_files collection
        audio_file_doc = {
            "filename": file.filename,
            "gridfs_id": gridfs_id,
            "user_id": current_user_id,
            "file_size": len(content),
            "content_type": file.content_type,
            "uploaded_at": datetime.utcnow(),
            "status": "uploaded"  # uploaded, processing, normalized, error
        }
        
        result = await db["audio_files"].insert_one(audio_file_doc)
        file_id = result.inserted_id
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "file_id": str(file_id),
            "filename": file.filename,
            "size_bytes": len(content)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Upload failed: {str(e)}"}
        )

@router.get("/uploads")
async def list_recent_uploads(limit: int = 10, current_user_id: str = Depends(get_current_user)):
    """
    Get list of recently uploaded files for the current user
    """
    try:
        db = await get_db()
        
        # Get uploads for current user
        cursor = db["audio_files"].find(
            {"user_id": current_user_id}
        ).sort("uploaded_at", -1).limit(limit)
        
        uploads = []
        async for doc in cursor:
            uploads.append({
                "id": str(doc["_id"]),
                "filename": doc["filename"],
                "size_bytes": doc.get("file_size", 0),
                "uploaded_at": doc["uploaded_at"].isoformat(),
                "status": doc.get("status", "uploaded")
            })
        
        return {
            "status": "success",
            "uploads": uploads,
            "count": len(uploads)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch uploads: {str(e)}"}
        )

@router.get("/user-files")
async def get_user_files(request: Request, limit: int = 50, current_user_id: str = Depends(get_current_user)):
    """
    Get all uploaded files for the current user with detailed information
    """
    try:
        db = await get_db()
        
        # Get all files for current user
        cursor = db["audio_files"].find(
            {"user_id": current_user_id}
        ).sort("uploaded_at", -1).limit(limit)
        
        files = []
        async for doc in cursor:
            # Get base URL for any links
            base_url = str(request.base_url).rstrip('/') if request else "http://localhost:8000"
            
            file_info = {
                "id": str(doc["_id"]),
                "filename": doc["filename"],
                "size_bytes": doc.get("file_size", 0),
                "uploaded_at": doc["uploaded_at"].isoformat(),
                "status": doc.get("status", "uploaded"),
                "stream_url": f"{base_url}/audio/stream-upload/{doc['_id']}",
            }
            
            # Add normalization info if available
            if doc.get("status") == "normalized":
                file_info.update({
                    "normalized": True,
                    "target_lufs": doc.get("target_lufs"),
                    "final_lufs": doc.get("final_lufs"),
                    "normalized_at": doc.get("normalized_at").isoformat() if doc.get("normalized_at") else None
                })
            
            files.append(file_info)
        
        return {
            "status": "success",
            "files": files,
            "count": len(files)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch user files: {str(e)}"}
        )

@router.get("/stream-upload/{file_id}")
async def stream_uploaded_file(file_id: str, request: Request, token: str = None):
    """
    Stream an uploaded (original) audio file with range support for seeking
    """
    try:
        # Get current user with flexible authentication
        current_user_id = await get_current_user_flexible(request, token)
        db = await get_db()
        bucket = AsyncIOMotorGridFSBucket(db)
        

        from bson import ObjectId
        try:
            file_doc = await db["audio_files"].find_one({"_id": ObjectId(file_id)})
        except Exception as e:
            print(f"Error converting file_id to ObjectId: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid file id: {file_id}"}
            )
        if not file_doc:
            return JSONResponse(
                status_code=404,
                content={"error": "File not found"}
            )
        
        # Check if the file belongs to the current user
        if file_doc.get("user_id") != current_user_id:
            return JSONResponse(
                status_code=403,
                content={"error": "You can only stream your own files"}
            )
        
        # Get the GridFS file using the gridfs_id field
        gridfs_id = file_doc.get("gridfs_id")
        if not gridfs_id:
            return JSONResponse(
                status_code=404,
                content={"error": "File data not found in storage"}
            )

        try:
            # Get file info from GridFS using gridfs_id
            grid_out = await bucket.open_download_stream(gridfs_id)
            file_size = grid_out.length

            # Determine content type
            filename = file_doc.get("filename", "audio.mp3")
            file_extension = filename.split(".")[-1].lower()
            content_type_map = {
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "flac": "audio/flac",
                "m4a": "audio/mp4",
                "aac": "audio/aac",
                "ogg": "audio/ogg"
            }
            content_type = content_type_map.get(file_extension, "audio/mpeg")

            # Handle range requests for seeking
            range_header = request.headers.get('range')
            if range_header:
                byte_start = 0
                byte_end = file_size - 1

                # Parse range header
                if range_header.startswith('bytes='):
                    range_value = range_header[6:]
                    if '-' in range_value:
                        start_str, end_str = range_value.split('-', 1)
                        if start_str:
                            byte_start = int(start_str)
                        if end_str:
                            byte_end = int(end_str)

                # Ensure valid range
                if byte_end >= file_size:
                    byte_end = file_size - 1
                if byte_start < 0:
                    byte_start = 0
                if byte_start > byte_end:
                    return JSONResponse(
                        status_code=416,
                        content={"error": "Range not satisfiable"}
                    )

                content_length = byte_end - byte_start + 1

                # Read the requested range
                await grid_out.seek(byte_start)
                chunk_data = await grid_out.read(content_length)

                headers = {
                    'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(content_length),
                    'Content-Type': content_type,
                }

                from fastapi.responses import Response
                return Response(content=chunk_data, status_code=206, headers=headers)

            else:
                # Stream the entire file
                import io
                file_stream = io.BytesIO()
                await bucket.download_to_stream(gridfs_id, file_stream)
                file_data = file_stream.getvalue()

                headers = {
                    'Content-Length': str(file_size),
                    'Accept-Ranges': 'bytes',
                    'Content-Type': content_type,
                }

                from fastapi.responses import Response
                return Response(
                    content=file_data,
                    media_type=content_type,
                    headers=headers
                )

        except Exception as e:
            return JSONResponse(
                status_code=404,
                content={"error": f"Could not retrieve file: {str(e)}"}
            )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to stream file: {str(e)}"}
        )
