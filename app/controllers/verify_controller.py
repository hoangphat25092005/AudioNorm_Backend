from fastapi import APIRouter, HTTPException
from app.config.database import get_db
from bson import ObjectId

router = APIRouter()

@router.get("/verify-email")
async def verify_email(token: str):
    db = await get_db()
    user = await db.users.find_one({"verification_token": token})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token.")
    if user.get("is_verified"):
        return {"message": "Email already verified."}
    await db.users.update_one({"_id": ObjectId(user["_id"])}, {"$set": {"is_verified": True, "verification_token": None}})
    return {"message": "Email verified successfully."}
