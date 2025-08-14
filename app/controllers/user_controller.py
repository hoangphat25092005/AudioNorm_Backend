from fastapi import APIRouter, HTTPException, Depends
from app.models.user import UserResponse
from app.services.user_service import UserService
from app.config.jwt_dependency import get_current_user
from typing import List

router = APIRouter()

@router.get("/profile", response_model=UserResponse)
async def get_current_user_profile(current_user_id: str = Depends(get_current_user)):
    """Get the current user's profile based on JWT token"""
    user = await UserService.get_user_by_id(current_user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    user = await UserService.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
