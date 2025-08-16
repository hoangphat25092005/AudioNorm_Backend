from fastapi import APIRouter
from app.models.reset_password import ForgotPasswordRequest, ResetPasswordRequest
from app.services.reset_password_service import ResetPasswordService

router = APIRouter()

@router.post("/forgot-password")
async def forgot_password(data: ForgotPasswordRequest):
    return await ResetPasswordService.request_reset(data)

@router.post("/reset-password")
async def reset_password(data: ResetPasswordRequest):
    return await ResetPasswordService.reset_password(data)
