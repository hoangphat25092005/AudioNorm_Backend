from pydantic import BaseModel, Field
from typing import Optional

#rating by star

class Feedback(BaseModel):
    feedback_text: str = Field(..., description="Text of the feedback")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating given by the user (1 to 5 stars)")
    created_at: Optional[str] = Field(None, description="Timestamp when the feedback was created")