#adding controller base on feedback.service.py and feedback.model.py
from fastapi import APIRouter, HTTPException, Depends
from app.models.feedback_model import Feedback
from app.services.feedback_service import FeedbackService
from app.config.jwt_dependency import get_current_user

router = APIRouter()

@router.post("/submit", response_model=dict, status_code=201)
async def submit_feedback(feedback: Feedback, user_id: str = Depends(get_current_user)):
    """
    Submit feedback for the application.
    """
    try:
        result = await FeedbackService.submit_feedback(feedback, user_id)
        return {"message": "Feedback submitted successfully", "feedback": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
