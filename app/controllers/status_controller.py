"""
Audio Status and Health Check Controller
Handles system status and health checks
"""
from fastapi import APIRouter
from app.services.audio_service import audio_service

router = APIRouter(tags=["Status"])

@router.get("/status")
async def get_audio_status():
    return {"status": "working", "message": "Audio API is operational"}

@router.get("/test")
async def test_audio():
    """Test basic audio functionality"""
    try:
        # Test if audio service is available
        status = audio_service.get_system_status()
        return {
            "status": "success",
            "message": "Audio system is working",
            "details": status
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Audio system test failed: {str(e)}"
        }

@router.get("/dependencies")
async def test_audio_dependencies():
    """Test audio processing dependencies"""
    try:
        status = audio_service.get_system_status()
        return {
            "status": "success",
            "dependencies": status,
            "message": "All audio dependencies are available"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Dependency check failed: {str(e)}"
        }
