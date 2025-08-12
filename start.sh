#!/bin/bash

# Production startup script for Render deployment

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run database migrations/setup if needed
echo "Setting up database connection..."

# Start the application
echo "Starting AudioNorm Backend..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
