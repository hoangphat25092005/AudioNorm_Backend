"""
Audio Export Controller
Handles all audio export and download operations
"""
from fastapi import APIRouter, Request, Depends, Response, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId
from datetime import datetime
import os
import io
import asyncio
import urllib.parse

from app.config.database import get_db
from app.config.jwt_dependency import get_current_user, get_current_user_flexible

router = APIRouter(tags=["Export"])

@router.get("/export-all")
async def export_all_files(
    format: str = "zip", 
    limit: int = 100, 
    current_user_id: str = Depends(get_current_user)
):
    """
    Export all normalized audio files as a zip archive.
    
    Parameters:
    - format: Export format (currently only 'zip' is supported)
    - limit: Maximum number of files to include (default: 100)
    
    Returns a zip file containing all normalized audio files.
    """
    if format != "zip":
        return JSONResponse(
            status_code=400,
            content={"error": "Only 'zip' format is currently supported"}
        )
    
    try:
        db = await get_db()
        bucket = AsyncIOMotorGridFSBucket(db)
        
        # Find all normalized files for the current user
        cursor = db["audio_files"].find({
            "user_id": current_user_id,
            "status": "normalized",
            "ready_to_download": True
        }).limit(limit)
        
        files = await cursor.to_list(length=limit)
        
        if not files:
            return JSONResponse(
                status_code=404,
                content={"error": "No normalized files found for this user"}
            )
        
        # Create zip file in memory
        import io
        import zipfile
        from datetime import datetime
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_doc in files:
                try:
                    # Get the normalized file from GridFS
                    gridfs_id = file_doc.get("normalized_gridfs_id")
                    if not gridfs_id:
                        print(f"Warning: No normalized_gridfs_id for file {file_doc['_id']}")
                        continue
                    
                    # Download file data
                    file_stream = io.BytesIO()
                    await bucket.download_to_stream(gridfs_id, file_stream)
                    file_data = file_stream.getvalue()
                    
                    if file_data:
                        # Use the clean normalized filename
                        filename = file_doc.get("normalized_filename", f"normalized_{file_doc['_id']}.mp3")
                        zip_file.writestr(filename, file_data)
                        print(f"Added to zip: {filename}")
                    
                except Exception as file_error:
                    print(f"Error adding file {file_doc['_id']} to zip: {str(file_error)}")
                    continue
        
        zip_buffer.seek(0)
        zip_data = zip_buffer.getvalue()
        
        if not zip_data:
            return JSONResponse(
                status_code=404,
                content={"error": "No valid normalized files could be exported"}
            )
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"normalized_audio_{timestamp}.zip"
        
        return StreamingResponse(
            io.BytesIO(zip_data),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
        
    except Exception as e:
        print(f"Error in export_all_files: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Export failed: {str(e)}"}
        )

@router.get("/export/{file_id}")
async def export_audio_file(
    file_id: str, 
    request: Request,
    token: str = None
):
    """
    Export/download a specific normalized audio file by its ID
    
    Parameters:
    - file_id: The ID of the normalized audio file to download
    - token: Optional query parameter for authentication (alternative to Authorization header)
    
    Returns the audio file for download.
    """
    # Get current user with flexible auth
    current_user_id = await get_current_user_flexible(request, token)
    try:
        db = await get_db()
        bucket = AsyncIOMotorGridFSBucket(db)
        
        # Look for the file in the audio_normalizations collection
        # Only fetch the fields we need to reduce query time
        projection = {
            "user_id": 1,
            "gridfs_id": 1, 
            "normalized_filename": 1,
            "original_filename": 1,
            "target_lufs": 1,
            "final_lufs": 1
        }
        normalized_doc = await db["audio_normalizations"].find_one(
            {"_id": ObjectId(file_id)}, 
            projection
        )
        
        if not normalized_doc:
            return JSONResponse(
                status_code=404,
                content={"error": f"Normalized file with ID {file_id} not found. This endpoint is for exporting normalized files only."}
            )
        
        # Check if the file belongs to the current user
        if normalized_doc.get("user_id") != current_user_id:
            return JSONResponse(
                status_code=403,
                content={"error": "You can only download your own files"}
            )
        
        # Get the GridFS ID for the normalized file
        gridfs_id = normalized_doc.get("gridfs_id")
        if not gridfs_id:
            return JSONResponse(
                status_code=404,
                content={"error": "Normalized file data not found in storage"}
            )
        
        # Create user-friendly filename with LUFS info
        original_filename = normalized_doc.get("original_filename", f"normalized_{file_id}")
        target_lufs = normalized_doc.get("target_lufs")
        
        # Extract base name and extension from original filename
        if "." in original_filename:
            base_name = ".".join(original_filename.split(".")[:-1])
            extension = original_filename.split(".")[-1]
        else:
            base_name = original_filename
            extension = "wav"  # Default extension
        
        # Create filename with LUFS info: "Song Name (-23 LUFS).mp3"
        if target_lufs is not None:
            filename = f"{base_name} ({target_lufs} LUFS).{extension}"
        else:
            filename = f"{base_name} (normalized).{extension}"
        
        # Determine content type based on file extension
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
        
        # Stream the file from GridFS
        try:
            # Get file size for Content-Length header (required for browser progress)
            try:
                grid_file_info = await bucket.find({"_id": gridfs_id}).to_list(length=1)
                file_size = grid_file_info[0].length if grid_file_info else None
            except:
                file_size = None
            
            # Create streaming generator that yields chunks immediately
            async def file_streamer():
                grid_out = await bucket.open_download_stream(gridfs_id)
                chunk_size = 4096  # 4KB chunks for more responsive progress updates
                try:
                    while True:
                        chunk = await grid_out.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
                        # Small delay to prevent overwhelming the connection
                        await asyncio.sleep(0)  # Yield control to event loop
                except Exception as e:
                    print(f"Error during streaming: {e}")
                    raise
                finally:
                    # Ensure proper cleanup
                    grid_out = None
            
            # Handle Unicode characters in filename for HTTP header
            try:
                # Try to encode as ASCII first
                filename.encode('ascii')
                safe_filename = filename
            except UnicodeEncodeError:
                # If filename contains Unicode characters, use RFC 5987 encoding
                encoded_filename = urllib.parse.quote(filename, safe='')
                safe_filename = f"UTF-8''{encoded_filename}"
            
            # Create proper Content-Disposition header
            if safe_filename == filename:
                # ASCII filename - use simple format
                content_disposition = f'attachment; filename="{safe_filename}"'
            else:
                # Unicode filename - use RFC 5987 format
                content_disposition = f"attachment; filename*={safe_filename}"
            
            # Update download stats (keep it simple and synchronous)
            try:
                await db["audio_normalizations"].update_one(
                    {"_id": ObjectId(file_id)},
                    {
                        "$inc": {"download_count": 1},
                        "$set": {"last_downloaded": datetime.utcnow()}
                    }
                )
            except Exception as e:
                # Don't let stats update errors affect the download
                print(f"Warning: Could not update download stats: {e}")
            
            # Create headers for browser download with progress
            headers = {
                "Content-Disposition": content_disposition,
                "Content-Type": content_type,
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Accept-Ranges": "bytes",
                "Transfer-Encoding": "chunked"
            }
            
            # Add Content-Length if we have file size (crucial for browser progress)
            if file_size:
                headers["Content-Length"] = str(file_size)
                # Remove Transfer-Encoding when we have Content-Length
                headers.pop("Transfer-Encoding", None)
            
            # Return streaming response - browser will show progress with Content-Length
            return StreamingResponse(
                file_streamer(),
                headers=headers,
                media_type=content_type
            )
            
        except Exception as stream_error:
            print(f"Error streaming file {file_id}: {str(stream_error)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Could not stream file: {str(stream_error)}"}
            )
        
    except Exception as e:
        print(f"Error in export_audio_file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Export failed: {str(e)}"}
        )

@router.get("/stream/{file_id}")
async def stream_audio_file(file_id: str, request: Request, current_user_id: str = Depends(get_current_user)):
    """
    Stream an audio file for playback (supports range requests for seeking)
    
    Parameters:
    - file_id: The ID of the audio file to stream
    
    Returns the audio file with proper streaming headers.
    """
    try:
        db = await get_db()
        bucket = AsyncIOMotorGridFSBucket(db)
        
        # Find the file document
        file_doc = await db["audio_files"].find_one({"_id": ObjectId(file_id)})
        
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check user permissions
        if file_doc.get("user_id") != current_user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Determine which file to stream (normalized or original)
        if file_doc.get("status") == "normalized" and file_doc.get("normalized_gridfs_id"):
            gridfs_id = file_doc.get("normalized_gridfs_id")
            filename = file_doc.get("normalized_filename", "audio.mp3")
        else:
            gridfs_id = file_doc.get("gridfs_id")
            filename = file_doc.get("filename", "audio.mp3")
        
        if not gridfs_id:
            raise HTTPException(status_code=404, detail="File data not found")
        
        # Get file info from GridFS
        try:
            grid_out = await bucket.open_download_stream(gridfs_id)
            file_size = grid_out.length
            
            # Determine content type
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
                    raise HTTPException(status_code=416, detail="Range not satisfiable")
                
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
                
                return StreamingResponse(
                    io.BytesIO(file_data),
                    media_type=content_type,
                    headers=headers
                )
                
        except Exception as stream_error:
            print(f"Error streaming file {file_id}: {str(stream_error)}")
            raise HTTPException(status_code=500, detail=f"Streaming error: {str(stream_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in stream_audio_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@router.delete("/delete/{file_id}")
async def delete_audio_file(file_id: str, current_user_id: str = Depends(get_current_user)):
    """
    Delete an audio file (both original and normalized if they exist)
    
    Parameters:
    - file_id: The ID of the audio file to delete
    
    Returns confirmation of deletion.
    """
    try:
        db = await get_db()
        bucket = AsyncIOMotorGridFSBucket(db)
        
        # Find the file document
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
                content={"error": "You can only delete your own files"}
            )
        
        files_deleted = []
        
        # Delete original file from GridFS if it exists
        original_gridfs_id = file_doc.get("gridfs_id")
        if original_gridfs_id:
            try:
                await bucket.delete(original_gridfs_id)
                files_deleted.append("original")
                print(f"Deleted original file from GridFS: {original_gridfs_id}")
            except Exception as e:
                print(f"Warning: Could not delete original file from GridFS: {str(e)}")
        
        # Delete normalized file from GridFS if it exists
        normalized_gridfs_id = file_doc.get("normalized_gridfs_id")
        if normalized_gridfs_id:
            try:
                await bucket.delete(normalized_gridfs_id)
                files_deleted.append("normalized")
                print(f"Deleted normalized file from GridFS: {normalized_gridfs_id}")
            except Exception as e:
                print(f"Warning: Could not delete normalized file from GridFS: {str(e)}")
        
        # Delete the document from the collection
        result = await db["audio_files"].delete_one({"_id": ObjectId(file_id)})
        
        if result.deleted_count == 1:
            return {
                "status": "success",
                "message": f"File {file_id} deleted successfully",
                "files_deleted": files_deleted,
                "filename": file_doc.get("filename", "unknown")
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to delete file document"}
            )
        
    except Exception as e:
        print(f"Error in delete_audio_file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Deletion failed: {str(e)}"}
        )

@router.delete("/cleanup-orphaned")
async def cleanup_orphaned_files(current_user_id: str = Depends(get_current_user)):
    """
    Clean up orphaned GridFS files that don't have corresponding database records
    (Admin functionality - could be restricted further)
    """
    try:
        db = await get_db()
        bucket = AsyncIOMotorGridFSBucket(db)
        
        # Get all GridFS files
        gridfs_files = []
        async for file_info in bucket.find():
            gridfs_files.append(file_info._id)
        
        # Get all referenced GridFS IDs from audio_files collection
        referenced_ids = set()
        async for doc in db["audio_files"].find({"user_id": current_user_id}):
            if doc.get("gridfs_id"):
                referenced_ids.add(doc["gridfs_id"])
            if doc.get("normalized_gridfs_id"):
                referenced_ids.add(doc["normalized_gridfs_id"])
        
        # Find orphaned files
        orphaned_files = [file_id for file_id in gridfs_files if file_id not in referenced_ids]
        
        # Delete orphaned files
        deleted_count = 0
        for file_id in orphaned_files:
            try:
                await bucket.delete(file_id)
                deleted_count += 1
                print(f"Deleted orphaned file: {file_id}")
            except Exception as e:
                print(f"Error deleting orphaned file {file_id}: {str(e)}")
        
        return {
            "status": "success",
            "message": f"Cleanup completed",
            "total_gridfs_files": len(gridfs_files),
            "referenced_files": len(referenced_ids),
            "orphaned_files_found": len(orphaned_files),
            "orphaned_files_deleted": deleted_count
        }
        
    except Exception as e:
        print(f"Error in cleanup_orphaned_files: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Cleanup failed: {str(e)}"}
        )
