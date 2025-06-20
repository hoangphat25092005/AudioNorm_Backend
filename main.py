from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.database import init_db, close_db
from app.controllers import auth_controller

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

# Database lifecycle
@app.on_event("startup")
async def startup_event():
    await init_db()

@app.on_event("shutdown")
async def shutdown_event():
    await close_db()

# Include routers
app.include_router(auth_controller.router, prefix="/auth", tags=["Authentication"])

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to AudioNorm API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)