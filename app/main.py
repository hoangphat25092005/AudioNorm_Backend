from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config.database import init_db, close_db
from app.controllers import auth_controller, feedback_controller, user_controller, audio_controller
import os

app = FastAPI(
    title="AudioNorm API",
    description="Audio Normalization API with user authentication and email notifications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for production
allowed_origins = [
    "http://localhost:3000",  # React development
    "http://localhost:8000",  # Local FastAPI
    "https://audionorm-backend.onrender.com",  # Production backend
    "https://audionorm-frontend.onrender.com",  # Your frontend on Render
]

# In production, be more restrictive with CORS
if os.getenv("ENVIRONMENT") == "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
else:
    # Development - allow all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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

# Database lifecycle
@app.on_event("startup")
async def startup_event():
    await init_db()

@app.on_event("shutdown")
async def shutdown_event():
    await close_db()

# Include routers
app.include_router(auth_controller.router, prefix="/auth", tags=["Authentication"])
app.include_router(feedback_controller.router, prefix="/feedback", tags=["Feedback"])
app.include_router(user_controller.router, prefix="/users", tags=["Users"])
app.include_router(audio_controller.router, prefix="/audio", tags=["Audio"])

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to AudioNorm API",
        "version": "1.0.0",
        "status": "healthy",
        "docs_url": "/docs",
        "environment": os.getenv("ENVIRONMENT", "development")
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
