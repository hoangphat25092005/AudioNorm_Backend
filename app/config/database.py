from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL") or os.getenv("MONGO_URI")
client = None
db = None

async def init_db():
    global client, db
    if client is None:
        client = AsyncIOMotorClient(MONGODB_URL, server_api=ServerApi('1'))
        db = client.audionorm
        try:
            await client.admin.command('ping')
            print("Successfully connected to MongoDB!")
        except Exception as e:
            print(f"Unable to connect to MongoDB: {e}")
            raise

async def get_db():
    if db is None:
        await init_db()
    return db

async def close_db():
    global client
    if client is not None:
        await client.close()
        client = None
