import asyncio
import sys
import os
from bson import ObjectId
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config.database import get_db

async def check_user_lookup():
    db = await get_db()
    
    # Get a sample feedback
    feedback = await db['feedbacks'].find_one({})
    if feedback:
        print(f"Sample feedback user_id: {feedback.get('user_id')} (type: {type(feedback.get('user_id'))})")
        
        # Try to find the user manually
        user_id = feedback.get('user_id')
        
        # Try as string first
        user_by_string = await db['users'].find_one({"_id": user_id})
        print(f"User found by string lookup: {user_by_string}")
        
        # Try as ObjectId
        try:
            user_object_id = ObjectId(user_id)
            user_by_objectid = await db['users'].find_one({"_id": user_object_id})
            print(f"User found by ObjectId lookup: {user_by_objectid}")
        except Exception as e:
            print(f"Error converting to ObjectId: {e}")
        
        # Test the aggregation pipeline manually
        print("\nTesting aggregation pipeline:")
        pipeline = [
            {"$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "_id",
                "as": "user"
            }},
            {"$limit": 1}
        ]
        
        result = await db['feedbacks'].aggregate(pipeline).to_list(1)
        print(f"Aggregation result: {result}")

if __name__ == "__main__":
    asyncio.run(check_user_lookup())
