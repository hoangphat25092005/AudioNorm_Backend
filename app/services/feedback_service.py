#adding feedback service base on feedback.model.py with mongodb
from app.models.feedback_model import Feedback, FeedbackResponse, FeedbackWithResponses, FeedbackResponseWithUser
from app.config.database import get_db
from app.services.email_service import EmailService
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import HTTPException
from datetime import datetime
from bson import ObjectId
from typing import List
import logging

logger = logging.getLogger(__name__)

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

        # Send confirmation email to the user
        try:
            # Validate user_id before converting to ObjectId
            user_object_id = ObjectId(user_id)
            user = await db['users'].find_one({"_id": user_object_id})
            if user and user.get('email'):
                email_result = await EmailService.send_feedback_confirmation(
                    user_email=user['email'],
                    username=user.get('username', 'User'),
                    feedback_content=feedback.feedback_text,
                    feedback_id=str(result.inserted_id)
                )
                logger.info(f"Email confirmation result: {email_result}")
        except Exception as e:
            logger.warning(f"Failed to send feedback confirmation email: {str(e)}")
            # Don't fail feedback submission if email fails

        return feedback_dict
    
    @staticmethod
    async def submit_feedback_response(response: FeedbackResponse, user_id: str) -> dict:
        """Submit a response to existing feedback and send email notification"""
        db = await get_db()
        
        # Validate response text
        if not response.response_text:
            raise HTTPException(status_code=400, detail="Response text is required")
        if len(response.response_text.split()) < 3:
            raise HTTPException(status_code=400, detail="Response must be at least 3 words long")
        
        # Check if feedback exists
        try:
            feedback_obj_id = ObjectId(response.feedback_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid feedback ID")
            
        feedback = await db['feedbacks'].find_one({"_id": feedback_obj_id})
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        # Get feedback author details
        try:
            feedback_user_id = ObjectId(feedback['user_id'])
            feedback_author = await db['users'].find_one({"_id": feedback_user_id})
        except Exception:
            feedback_author = None
        
        # Get response author details
        try:
            response_user_id = ObjectId(user_id)
            response_author = await db['users'].find_one({"_id": response_user_id})
        except Exception:
            response_author = None
        
        # Create response document
        response_dict = response.dict()
        response_dict['user_id'] = user_id
        response_dict['created_at'] = response.created_at or str(datetime.utcnow())
        response_dict['feedback_id'] = feedback_obj_id
        
        # Insert response
        result = await db['feedback_responses'].insert_one(response_dict)
        response_dict["_id"] = str(result.inserted_id)
        response_dict["feedback_id"] = response.feedback_id  # Keep as string for return
        
        # Send email notification to feedback author (if not responding to own feedback)
        try:
            if (feedback_author and 
                response_author and 
                str(feedback_author['_id']) != user_id and
                feedback_author.get('email')):
                
                await EmailService.send_feedback_notification(
                    feedback_author_email=feedback_author['email'],
                    feedback_author_name=feedback_author.get('username', 'User'),
                    responder_name=response_author.get('username', 'Someone'),
                    responder_email=response_author.get('email', ''),
                    feedback_title=feedback['feedback_text'][:50] + "..." if len(feedback['feedback_text']) > 50 else feedback['feedback_text'],
                    response_content=response.response_text,
                    feedback_id=response.feedback_id
                )
        except Exception as e:
            logger.error(f"Failed to send feedback response email: {str(e)}")
            # Don't fail response submission if email fails
        
        return response_dict
    
    @staticmethod
    async def get_feedback_with_responses(feedback_id: str) -> FeedbackWithResponses:
        """Get feedback with all its responses"""
        db = await get_db()
        
        try:
            feedback_obj_id = ObjectId(feedback_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid feedback ID")
        
        # Get feedback with user details
        feedback_pipeline = [
            {"$match": {"_id": feedback_obj_id}},
            {"$addFields": {
                "user_object_id": {"$toObjectId": "$user_id"}
            }},
            {"$lookup": {
                "from": "users",
                "localField": "user_object_id",
                "foreignField": "_id",
                "as": "user"
            }},
            {"$unwind": "$user"}
        ]
        
        feedback_result = await db['feedbacks'].aggregate(feedback_pipeline).to_list(1)
        if not feedback_result:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        feedback = feedback_result[0]
        
        # Get responses with user details
        response_pipeline = [
            {"$match": {"feedback_id": feedback_obj_id}},
            {"$addFields": {
                "user_object_id": {"$toObjectId": "$user_id"}
            }},
            {"$lookup": {
                "from": "users",
                "localField": "user_object_id",
                "foreignField": "_id",
                "as": "user"
            }},
            {"$unwind": "$user"},
            {"$sort": {"created_at": 1}}
        ]
        
        responses = await db['feedback_responses'].aggregate(response_pipeline).to_list(None)
        
        response_list = [
            FeedbackResponseWithUser(
                id=str(resp["_id"]),
                response_text=resp["response_text"],
                feedback_id=feedback_id,
                user_id=str(resp["user_id"]),
                user_name=resp["user"].get("username"),
                user_email=resp["user"].get("email"),
                created_at=resp["created_at"]
            )
            for resp in responses
        ]
        
        return FeedbackWithResponses(
            id=str(feedback["_id"]),
            feedback_text=feedback["feedback_text"],
            rating=feedback.get("rating"),
            user_id=str(feedback["user_id"]),
            user_name=feedback["user"].get("username"),
            user_email=feedback["user"].get("email"),
            created_at=feedback["created_at"],
            responses=response_list,
            response_count=len(response_list)
        )
    
    @staticmethod
    async def get_all_feedback_with_responses() -> List[FeedbackWithResponses]:
        """Get all feedback with response counts"""
        db = await get_db()
        
        # Aggregate feedback with user details and response counts
        pipeline = [
            {"$addFields": {
                "user_object_id": {"$toObjectId": "$user_id"}
            }},
            {"$lookup": {
                "from": "users",
                "localField": "user_object_id",
                "foreignField": "_id",
                "as": "user"
            }},
            {"$unwind": "$user"},
            {"$lookup": {
                "from": "feedback_responses",
                "localField": "_id",
                "foreignField": "feedback_id",
                "as": "responses"
            }},
            {"$addFields": {
                "response_count": {"$size": "$responses"}
            }},
            {"$sort": {"created_at": -1}}
        ]
        
        feedback_list = await db['feedbacks'].aggregate(pipeline).to_list(None)
        
        return [
            FeedbackWithResponses(
                id=str(feedback["_id"]),
                feedback_text=feedback["feedback_text"],
                rating=feedback.get("rating"),
                user_id=str(feedback["user_id"]),
                user_name=feedback["user"].get("username"),
                user_email=feedback["user"].get("email"),
                created_at=feedback["created_at"],
                responses=[],  # Don't load full responses in list view
                response_count=feedback["response_count"]
            )
            for feedback in feedback_list
        ]
    
    @staticmethod
    async def get_user_feedback(user_id: str) -> List[FeedbackWithResponses]:
        """Get all feedback submitted by a specific user"""
        db = await get_db()
        
        try:
            user_object_id = ObjectId(user_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid user ID")
        
        pipeline = [
            {"$match": {"user_id": user_id}},  # Match as string since user_id is stored as string
            {"$addFields": {
                "user_object_id": {"$toObjectId": "$user_id"}
            }},
            {"$lookup": {
                "from": "users",
                "localField": "user_object_id",
                "foreignField": "_id",
                "as": "user"
            }},
            {"$unwind": "$user"},
            {"$lookup": {
                "from": "feedback_responses",
                "localField": "_id",
                "foreignField": "feedback_id",
                "as": "responses"
            }},
            {"$addFields": {
                "response_count": {"$size": "$responses"}
            }},
            {"$sort": {"created_at": -1}}
        ]
        
        feedback_list = await db['feedbacks'].aggregate(pipeline).to_list(None)
        
        return [
            FeedbackWithResponses(
                id=str(feedback["_id"]),
                feedback_text=feedback["feedback_text"],
                rating=feedback.get("rating"),
                user_id=str(feedback["user_id"]),
                user_name=feedback["user"].get("username"),
                user_email=feedback["user"].get("email"),
                created_at=feedback["created_at"],
                responses=[],
                response_count=feedback["response_count"]
            )
            for feedback in feedback_list
        ]