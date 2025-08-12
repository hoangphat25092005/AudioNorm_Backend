from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

#rating by star

class Feedback(BaseModel):
    feedback_text: str = Field(..., description="Text of the feedback")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating given by the user (1 to 5 stars)")
    created_at: Optional[str] = Field(None, description="Timestamp when the feedback was created")

# New models for feedback responses
class FeedbackResponse(BaseModel):
    response_text: str = Field(..., description="Text of the response to feedback")
    feedback_id: str = Field(..., description="ID of the feedback being responded to")
    created_at: Optional[str] = Field(None, description="Timestamp when the response was created")

class FeedbackWithResponses(BaseModel):
    id: str
    feedback_text: str
    rating: Optional[int]
    user_id: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    created_at: str
    responses: list = Field(default_factory=list, description="List of responses to this feedback")
    response_count: int = Field(default=0, description="Number of responses")

class FeedbackResponseWithUser(BaseModel):
    id: str
    response_text: str
    feedback_id: str
    user_id: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    created_at: str