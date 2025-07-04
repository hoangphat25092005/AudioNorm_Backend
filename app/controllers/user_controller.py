from fastapi import APIRouter, HTTPException, Depends
from app.models.user import UserResponse
from app.services.user_service import UserService
from typing import List

router = APIRouter()

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    user = await UserService.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        _id=str(user.id),
        email=user.email,
        username=user.username,
        profile_picture=user.profile_picture,
        auth_provider=user.auth_provider,
        created_at=user.created_at
    )
