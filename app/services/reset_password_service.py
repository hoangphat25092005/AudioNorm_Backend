import secrets
from datetime import datetime, timedelta
from fastapi import HTTPException
from app.config.database import get_db
from app.models.reset_password import ForgotPasswordRequest, ResetPasswordRequest
from app.services.email_service import EmailService
from app.services.auth_service import AuthService
import os

RESET_TOKEN_EXPIRE_HOURS = 1

class ResetPasswordService:
    @staticmethod
    async def request_reset(data: ForgotPasswordRequest):
        db = await get_db()
        user = await db.users.find_one({"email": data.email})
        if not user:
            # Don't reveal if email exists
            return {"message": "If your email is registered, a reset link has been sent."}
        # Only allow reset if user is verified
        is_verified = user.get("is_verified", False)
        if not is_verified:
            # Don't reveal if email exists or not verified
            return {"message": "If your email is registered and verified, a reset link has been sent."}
        token = secrets.token_urlsafe(32)
        expire = datetime.utcnow() + timedelta(hours=RESET_TOKEN_EXPIRE_HOURS)
        await db.users.update_one({"_id": user["_id"]}, {"$set": {"reset_token": token, "reset_token_expire": expire}})
        base_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        reset_url = f"{base_url}/reset-password?token={token}"
        await EmailService.send_reset_password_email(user["email"], user["username"], reset_url, is_verified=is_verified)
        return {"message": "If your email is registered and verified, a reset link has been sent."}

    @staticmethod
    async def reset_password(data: ResetPasswordRequest):
        db = await get_db()
        user = await db.users.find_one({"reset_token": data.token})
        if not user or not user.get("reset_token_expire"):
            raise HTTPException(status_code=400, detail="Invalid or expired token.")
        if datetime.utcnow() > user["reset_token_expire"]:
            raise HTTPException(status_code=400, detail="Token expired.")
        hashed = AuthService.get_password_hash(data.password)
        await db.users.update_one({"_id": user["_id"]}, {"$set": {"password": hashed}, "$unset": {"reset_token": "", "reset_token_expire": ""}})
        return {"message": "Password reset successful."}
