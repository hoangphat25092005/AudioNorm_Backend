#adding controller base on feedback.service.py and feedback.model.py
from fastapi import APIRouter, HTTPException, Depends
from app.models.feedback_model import Feedback, FeedbackResponse, FeedbackWithResponses
from app.services.feedback_service import FeedbackService
from app.config.jwt_dependency import get_current_user
from typing import List

router = APIRouter()

@router.post("/submit", response_model=dict, status_code=201)
async def submit_feedback(feedback: Feedback, user_id: str = Depends(get_current_user)):
    """
    Submit feedback for the application.
    Sends confirmation email to the user.
    """
    try:
        result = await FeedbackService.submit_feedback(feedback, user_id)
        return {"message": "Feedback submitted successfully", "feedback": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/respond", response_model=dict, status_code=201)
async def respond_to_feedback(response: FeedbackResponse, user_id: str = Depends(get_current_user)):
    """
    Respond to existing feedback.
    Sends email notification to the feedback author.
    """
    try:
        result = await FeedbackService.submit_feedback_response(response, user_id)
        return {"message": "Response submitted successfully", "response": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[FeedbackWithResponses])
async def get_all_feedback():
    """
    Get all feedback with response counts.
    Public endpoint - no authentication required.
    """
    try:
        return await FeedbackService.get_all_feedback_with_responses()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{feedback_id}", response_model=FeedbackWithResponses)
async def get_feedback_with_responses(feedback_id: str):
    """
    Get specific feedback with all its responses.
    Public endpoint - no authentication required.
    """
    try:
        return await FeedbackService.get_feedback_with_responses(feedback_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/my-feedback", response_model=List[FeedbackWithResponses])
async def get_my_feedback(user_id: str = Depends(get_current_user)):
    """
    Get all feedback submitted by the current user.
    Requires authentication.
    """
    try:
        return await FeedbackService.get_user_feedback(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
