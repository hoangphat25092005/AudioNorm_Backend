#adding feedback service base on feedback.model.py with mongodb
from app.models.feedback_model import Feedback
from app.config.database import get_db
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import HTTPException
from datetime import datetime

# How can i take JWT token from user and check whether they login or not

class FeedbackService:
    @staticmethod
    async def submit_feedback(feedback: Feedback, user_id: str) -> dict:
        # check whether they login or not
        db = await get_db()
        # feedback at least have user_id and feedback_text
        if not feedback.feedback_text:
            raise HTTPException(status_code=400, detail="Feedback text is required")
        # Feedback at least 3 words
        if len(feedback.feedback_text.split()) < 3:
            raise HTTPException(status_code=400, detail="Feedback must be at least 3 words long")
        #Rating is optional, but if provided, it must be between 1 and 5
        if feedback.rating is not None and (feedback.rating < 1 or feedback.rating > 5):
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        feedback_dict = feedback.dict()
        feedback_dict['user_id'] = user_id
        feedback_dict['created_at'] = feedback.created_at or str(datetime.utcnow())

        result = await db['feedbacks'].insert_one(feedback_dict)
        feedback_dict["_id"] = str(result.inserted_id)

        return feedback_dict