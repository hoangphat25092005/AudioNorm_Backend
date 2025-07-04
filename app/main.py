from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config.database import init_db, close_db
from app.controllers import auth_controller, feedback_controller, user_controller

app = FastAPI(
    title="AudioNorm API",
    description="Audio Normalization API with user authentication",
    version="1.0.0"
)

# Configure CORS
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

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to AudioNorm API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }
