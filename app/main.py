from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.config.database import init_db, close_db
from app.controllers import auth_controller, feedback_controller, user_controller
from app.controllers import status_controller, upload_controller, normalization_controller, export_controller, files_controller
import os
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await init_db()
        print("Application startup completed successfully")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        await close_db()
        print("Application shutdown completed successfully")
    except Exception as e:
        print(f"Error during shutdown: {e}")


app = FastAPI(
    title="AudioNorm API",
    description="Audio Normalization API with user authentication and email notifications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add session middleware for OAuth/session support
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "changeme-please-set-SECRET_KEY-env-var")
)

# --- Add main block for Uvicorn with dynamic port ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

# Configure CORS for production
allowed_origins = [
    "http://localhost:3000",  # React development
    "https://audionorm-frontend.onrender.com",
    
]



# Always allow both localhost and production frontends/backends for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://audionorm-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Include routers
app.include_router(auth_controller.router, prefix="/auth", tags=["Authentication"])
app.include_router(feedback_controller.router, prefix="/feedback", tags=["Feedback"])
app.include_router(user_controller.router, prefix="/users", tags=["Users"])

# Audio-related routers (broken down from monolithic controller)
app.include_router(status_controller.router, prefix="/audio", tags=["Audio Status"])
app.include_router(upload_controller.router, prefix="/audio", tags=["Audio Upload"])
app.include_router(normalization_controller.router, prefix="/audio", tags=["Audio Normalization"])
app.include_router(export_controller.router, prefix="/audio", tags=["Audio Export"])
app.include_router(files_controller.router, prefix="/audio", tags=["Audio Files"])
from app.controllers import verify_controller, reset_password_controller, review_audio_controller
app.include_router(verify_controller.router, prefix="/verify", tags=["Verify"])
app.include_router(reset_password_controller.router, prefix="/auth", tags=["Password Reset"])
app.include_router(review_audio_controller.router, tags=["Audio Preview"])
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to AudioNorm API",
        "version": "1.0.0",
        "status": "healthy",
        "docs_url": "/docs",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "features": {
            "authentication": "JWT-based user authentication",
            "feedback": "User feedback and response system with email notifications",
            "audio_normalization": "Complete audio normalization API with database storage and result tracking",
        },
        "endpoints": {
            "auth": "/auth/* - Authentication endpoints",
            "feedback": "/feedback/* - Feedback system",
            "users": "/users/* - User management", 
            "audio": {
                "/audio/status": "Audio service status and health checks",
                "/audio/test": "Test audio endpoints",
                "/audio/upload": "Upload audio files to GridFS",
                "/audio/files": "List and manage audio files",
                "/audio/files/normalized": "Get normalized files",
                "/audio/files/original": "Get original uploaded files",
                "/audio/normalize/{target_lufs}": "Normalize audio to target LUFS",
                "/audio/normalize-uploaded/{file_id}/{target_lufs}": "Normalize previously uploaded file",
                "/audio/export/{file_id}": "Download normalized audio files",
                "/audio/export-all": "Download all normalized files as ZIP",
                "/audio/stream/{file_id}": "Stream audio files for playback",
                "/audio/analyze": "Analyze audio properties without normalization"
            }
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {
        "status": "healthy",
        "timestamp": "2025-08-12T00:00:00Z",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }
