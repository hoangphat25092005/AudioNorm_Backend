import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config.database import get_db

async def check_database():
    db = await get_db()
    
    # Check if feedbacks collection exists and has data
    feedback_count = await db['feedbacks'].count_documents({})
    print(f"Total feedbacks in database: {feedback_count}")
    
    # List all feedbacks
    feedbacks = await db['feedbacks'].find({}).to_list(None)
    print(f"Feedbacks found: {len(feedbacks)}")
    
    for feedback in feedbacks:
        print(f"- ID: {feedback.get('_id')}")
        print(f"  Text: {feedback.get('feedback_text')}")
        print(f"  Rating: {feedback.get('rating')}")
        print(f"  User ID: {feedback.get('user_id')}")
        print(f"  Created: {feedback.get('created_at')}")
        print()
    
    # Check users collection
    user_count = await db['users'].count_documents({})
    print(f"Total users in database: {user_count}")
    
    users = await db['users'].find({}).to_list(None)
    for user in users:
        print(f"- User ID: {user.get('_id')}")
        print(f"  Username: {user.get('username')}")
        print(f"  Email: {user.get('email')}")
        print()

if __name__ == "__main__":
    asyncio.run(check_database())
