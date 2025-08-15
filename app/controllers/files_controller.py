"""
Audio Files Controller
Handles file listing, metadata, and file management operations
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId
from datetime import datetime, timedelta
from typing import Optional, List
import re
import urllib.parse

from app.config.database import get_db
from app.config.jwt_dependency import get_current_user

def sanitize_filename_for_http(filename: str) -> str:
    """
    Sanitize filename for HTTP headers to handle Unicode characters
    """
    if not filename:
        return ""
    
    try:
        # URL encode the filename to handle Unicode characters
        encoded = urllib.parse.quote(filename, safe='')
        return encoded
    except Exception:
        # Fallback: remove non-ASCII characters
        return re.sub(r'[^\x00-\x7F]+', '_', filename)

router = APIRouter(tags=["Files"])

@router.get("/files")
async def get_user_files(
    limit: int = Query(50, description="Maximum number of files to return"),
    offset: int = Query(0, description="Number of files to skip"),
    status: Optional[str] = Query(None, description="Filter by status: 'uploaded', 'normalized'"),
    current_user_id: str = Depends(get_current_user)
):
    """
    Get all files belonging to the current user with pagination and filtering
    
    Parameters:
    - limit: Maximum number of files to return (default: 50)
    - offset: Number of files to skip for pagination (default: 0)
    - status: Filter by file status ('uploaded', 'normalized', or None for all)
    
    Returns a list of user's files with metadata.
    """
    try:
        db = await get_db()
        
        # Build query filter
        query_filter = {"user_id": current_user_id}
        if status:
            if status == "uploaded":
                query_filter["status"] = {"$in": ["uploaded", None]}
            elif status == "normalized":
                query_filter["status"] = "normalized"
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid status filter. Use 'uploaded' or 'normalized'"}
                )
        
        # Get total count for pagination info
        total_count = await db["audio_files"].count_documents(query_filter)
        
        # Get files with pagination
        cursor = db["audio_files"].find(query_filter).sort("uploaded_at", -1).skip(offset).limit(limit)
        files = await cursor.to_list(length=limit)
        
        # Format files for response
        formatted_files = []
        for file_doc in files:
            # Create clean filename for display
            original_filename = file_doc.get("filename", "unknown.mp3")
            clean_name = original_filename
            if clean_name.startswith("uploaded_"):
                # Remove the uploaded_ prefix and timestamp
                parts = clean_name.split("_", 2)
                if len(parts) >= 3:
                    clean_name = parts[2]
            
            # Get user name from the document or use user_id as fallback
            user_display = file_doc.get("user_name", file_doc.get("user_id", "Unknown"))
            
            file_info = {
                "id": str(file_doc["_id"]),
                "filename": clean_name,
                "original_filename": original_filename,
                "normalized_filename": file_doc.get("normalized_filename"),
                "user_id": file_doc.get("user_id"),
                "user_name": user_display,
                "status": file_doc.get("status", "uploaded"),
                "uploaded_at": file_doc.get("uploaded_at"),
                "normalized_at": file_doc.get("normalized_at"),
                "ready_to_download": file_doc.get("ready_to_download", False),
                "file_size": file_doc.get("file_size"),
                "duration": file_doc.get("duration"),
                "download_count": file_doc.get("download_count", 0),
                "last_downloaded": file_doc.get("last_downloaded")
            }
            
            # Add normalization details if available
            if file_doc.get("status") == "normalized":
                file_info.update({
                    "original_lufs": file_doc.get("original_lufs"),
                    "target_lufs": file_doc.get("target_lufs"),
                    "final_lufs": file_doc.get("final_lufs"),
                    "normalization_method": file_doc.get("normalization_method"),
                    "processing_time_seconds": file_doc.get("processing_time_seconds")
                })
            
            formatted_files.append(file_info)
        
        return {
            "status": "success",
            "files": formatted_files,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "filter": {
                "status": status,
                "user_id": current_user_id
            }
        }
        
    except Exception as e:
        print(f"Error in get_user_files: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve files: {str(e)}"}
        )

@router.get("/files/normalized")
async def get_normalized_files(
    limit: int = Query(50, description="Maximum number of files to return"),
    offset: int = Query(0, description="Number of files to skip"),
    current_user_id: str = Depends(get_current_user)
):
    """
    Get all normalized files belonging to the current user
    
    Parameters:
    - limit: Maximum number of files to return (default: 50)
    - offset: Number of files to skip for pagination (default: 0)
    
    Returns a list of user's normalized files.
    """
    try:
        db = await get_db()
        
        # Query the audio_normalizations collection directly for normalized files
        query_filter = {
            "user_id": current_user_id
        }
        
        # Get total count
        total_count = await db["audio_normalizations"].count_documents(query_filter)
        
        # Get files with pagination
        cursor = db["audio_normalizations"].find(query_filter).sort("created_at", -1).skip(offset).limit(limit)
        files = await cursor.to_list(length=limit)
        
        # Format files for response
        formatted_files = []
        for norm_doc in files:
            # Get the clean filename for display
            original_filename = norm_doc.get("original_filename", "unknown.mp3")
            normalized_filename = norm_doc.get("normalized_filename", "normalized.wav")
            
            # Create clean name by removing prefixes
            clean_name = original_filename
            if clean_name.startswith("uploaded_"):
                # Remove the uploaded_ prefix and timestamp
                parts = clean_name.split("_", 2)
                if len(parts) >= 3:
                    clean_name = parts[2]
            
            file_info = {
                "id": str(norm_doc["_id"]),  # Use the normalized file ID for export
                "filename": clean_name,
                "original_filename": original_filename,
                "normalized_filename": normalized_filename,
                "user_id": norm_doc.get("user_id"),
                "user_name": norm_doc.get("user_id", "Unknown"),  # Could be enhanced with actual user lookup
                "uploaded_at": norm_doc.get("created_at"),  # Use creation time
                "normalized_at": norm_doc.get("created_at"),
                "file_size": norm_doc.get("file_size_bytes"),
                "duration": norm_doc.get("duration_seconds"),
                "download_count": 0,  # Not tracked in normalizations collection
                "last_downloaded": None,
                "original_lufs": norm_doc.get("original_lufs"),
                "target_lufs": norm_doc.get("target_lufs"),
                "final_lufs": norm_doc.get("final_lufs"),
                "normalization_method": norm_doc.get("normalization_method"),
                "ready_to_download": True  # All normalized files are ready
            }
            
            formatted_files.append(file_info)
        
        return {
            "status": "success",
            "normalized_files": formatted_files,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        print(f"Error in get_normalized_files: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve normalized files: {str(e)}"}
        )

@router.get("/files/original")
async def get_original_files(
    limit: int = Query(50, description="Maximum number of files to return"),
    offset: int = Query(0, description="Number of files to skip"),
    current_user_id: str = Depends(get_current_user)
):
    """
    Get all original (uploaded) files belonging to the current user
    
    Parameters:
    - limit: Maximum number of files to return (default: 50)
    - offset: Number of files to skip for pagination (default: 0)
    
    Returns a list of user's original uploaded files.
    """
    try:
        db = await get_db()
        
        # Query for original files (uploaded status or no status)
        query_filter = {
            "user_id": current_user_id,
            "$or": [
                {"status": "uploaded"},
                {"status": {"$exists": False}},
                {"status": None}
            ]
        }
        
        print(f"DEBUG: Searching for original files with query: {query_filter}")
        
        # Get total count
        total_count = await db["audio_files"].count_documents(query_filter)
        print(f"DEBUG: Found {total_count} original files for user {current_user_id}")
        
        # Get files with pagination
        cursor = db["audio_files"].find(query_filter).sort("uploaded_at", -1).skip(offset).limit(limit)
        files = await cursor.to_list(length=limit)
        
        # Format files for response
        formatted_files = []
        for file_doc in files:
            # Create clean filename for display
            original_filename = file_doc.get("filename", "unknown.mp3")
            clean_name = original_filename
            if clean_name.startswith("uploaded_"):
                # Remove the uploaded_ prefix and timestamp
                parts = clean_name.split("_", 2)
                if len(parts) >= 3:
                    clean_name = parts[2]
            
            # Get user name from the document or use user_id as fallback
            user_display = file_doc.get("user_name", file_doc.get("user_id", "Unknown"))
            
            file_info = {
                "id": str(file_doc["_id"]),
                "filename": clean_name,
                "original_filename": original_filename,
                "user_id": file_doc.get("user_id"),
                "user_name": user_display,
                "uploaded_at": file_doc.get("uploaded_at"),
                "file_size": file_doc.get("file_size"),
                "duration": file_doc.get("duration"),
                "can_normalize": True  # Original files can be normalized
            }
            
            formatted_files.append(file_info)
        
        return {
            "status": "success",
            "original_files": formatted_files,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        print(f"Error in get_original_files: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve original files: {str(e)}"}
        )

@router.get("/files/{file_id}")
async def get_file_details(file_id: str, current_user_id: str = Depends(get_current_user)):
    """
    Get detailed information about a specific file
    
    Parameters:
    - file_id: The ID of the file to get details for
    
    Returns detailed file information.
    """
    try:
        db = await get_db()
        
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
                content={"error": "You can only access your own files"}
            )
        
        # Create clean filename for display
        original_filename = file_doc.get("filename", "unknown.mp3")
        clean_name = original_filename
        if clean_name.startswith("uploaded_"):
            # Remove the uploaded_ prefix and timestamp
            parts = clean_name.split("_", 2)
            if len(parts) >= 3:
                clean_name = parts[2]
        
        # Get user name from the document or use user_id as fallback
        user_display = file_doc.get("user_name", file_doc.get("user_id", "Unknown"))
        
        # Build detailed file information
        file_details = {
            "id": str(file_doc["_id"]),
            "filename": clean_name,
            "original_filename": original_filename,
            "user_id": file_doc.get("user_id"),
            "user_name": user_display,
            "status": file_doc.get("status", "uploaded"),
            "uploaded_at": file_doc.get("uploaded_at"),
            "file_size": file_doc.get("file_size"),
            "duration": file_doc.get("duration"),
            "download_count": file_doc.get("download_count", 0),
            "last_downloaded": file_doc.get("last_downloaded"),
            "ready_to_download": file_doc.get("ready_to_download", False),
            "ip_address": file_doc.get("ip_address"),
            "user_agent": file_doc.get("user_agent")
        }
        
        # Add normalization details if available
        if file_doc.get("status") == "normalized":
            file_details.update({
                "normalized_filename": file_doc.get("normalized_filename"),
                "normalized_at": file_doc.get("normalized_at"),
                "original_lufs": file_doc.get("original_lufs"),
                "target_lufs": file_doc.get("target_lufs"),
                "final_lufs": file_doc.get("final_lufs"),
                "normalization_method": file_doc.get("normalization_method"),
                "processing_time_seconds": file_doc.get("processing_time_seconds"),
                "original_upload_id": file_doc.get("original_upload_id")
            })
        
        # Add GridFS information
        gridfs_info = {}
        if file_doc.get("gridfs_id"):
            gridfs_info["original_gridfs_id"] = str(file_doc["gridfs_id"])
        if file_doc.get("normalized_gridfs_id"):
            gridfs_info["normalized_gridfs_id"] = str(file_doc["normalized_gridfs_id"])
        
        if gridfs_info:
            file_details["storage"] = gridfs_info
        
        return {
            "status": "success",
            "file": file_details
        }
        
    except Exception as e:
        print(f"Error in get_file_details: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve file details: {str(e)}"}
        )

@router.get("/stats")
async def get_user_stats(current_user_id: str = Depends(get_current_user)):
    """
    Get statistics about the user's files
    
    Returns statistics including file counts, storage usage, etc.
    """
    try:
        db = await get_db()
        
        # Get file counts by status
        total_files = await db["audio_files"].count_documents({"user_id": current_user_id})
        uploaded_files = await db["audio_files"].count_documents({
            "user_id": current_user_id,
            "$or": [
                {"status": "uploaded"},
                {"status": {"$exists": False}},
                {"status": None}
            ]
        })
        normalized_files = await db["audio_files"].count_documents({
            "user_id": current_user_id,
            "status": "normalized"
        })
        
        # Get storage usage (sum of file sizes)
        pipeline = [
            {"$match": {"user_id": current_user_id}},
            {"$group": {
                "_id": None,
                "total_size": {"$sum": "$file_size"},
                "avg_size": {"$avg": "$file_size"}
            }}
        ]
        storage_stats = await db["audio_files"].aggregate(pipeline).to_list(length=1)
        total_storage = storage_stats[0].get("total_size", 0) if storage_stats else 0
        avg_file_size = storage_stats[0].get("avg_size", 0) if storage_stats else 0
        
        # Get download statistics
        download_pipeline = [
            {"$match": {"user_id": current_user_id, "status": "normalized"}},
            {"$group": {
                "_id": None,
                "total_downloads": {"$sum": "$download_count"},
                "avg_downloads": {"$avg": "$download_count"}
            }}
        ]
        download_stats = await db["audio_files"].aggregate(download_pipeline).to_list(length=1)
        total_downloads = download_stats[0].get("total_downloads", 0) if download_stats else 0
        avg_downloads = download_stats[0].get("avg_downloads", 0) if download_stats else 0
        
        # Get recent activity (files uploaded in last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_uploads = await db["audio_files"].count_documents({
            "user_id": current_user_id,
            "uploaded_at": {"$gte": seven_days_ago}
        })
        
        # Get most recent file
        recent_file_cursor = db["audio_files"].find({"user_id": current_user_id}).sort("uploaded_at", -1).limit(1)
        recent_files = await recent_file_cursor.to_list(length=1)
        last_upload = recent_files[0].get("uploaded_at") if recent_files else None
        
        return {
            "status": "success",
            "stats": {
                "files": {
                    "total": total_files,
                    "uploaded": uploaded_files,
                    "normalized": normalized_files,
                    "recent_uploads_7_days": recent_uploads
                },
                "storage": {
                    "total_bytes": int(total_storage) if total_storage else 0,
                    "total_mb": round(total_storage / 1024 / 1024, 2) if total_storage else 0,
                    "average_file_size_mb": round(avg_file_size / 1024 / 1024, 2) if avg_file_size else 0
                },
                "downloads": {
                    "total": int(total_downloads) if total_downloads else 0,
                    "average_per_file": round(avg_downloads, 1) if avg_downloads else 0
                },
                "activity": {
                    "last_upload": last_upload,
                    "recent_uploads_count": recent_uploads
                }
            }
        }
        
    except Exception as e:
        print(f"Error in get_user_stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve statistics: {str(e)}"}
        )
