from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import pathlib

# Load the .env file from the app directory
app_dir = pathlib.Path(__file__).parent.parent
env_path = app_dir / '.env'
load_dotenv(dotenv_path=env_path)

MONGODB_URL = os.getenv("MONGO_URI")  # Changed from MONGODB_URL to MONGO_URI
client = None
db = None

async def init_db():
    global client, db
    if client is None:
        print(f"Connecting to MongoDB at: {MONGODB_URL}")
        client = AsyncIOMotorClient(MONGODB_URL, server_api=ServerApi('1'))
        # Extract database name from URI or use default
        if MONGODB_URL and "/" in MONGODB_URL and MONGODB_URL.split("/")[-1]:
            db_name = MONGODB_URL.split("/")[-1].split("?")[0]  # Get DB name, remove query params
            db = client[db_name]
            print(f"Using database: {db_name}")
        else:
            db = client.audionorm_db  # Fallback to default name
            print("Using default database: audionorm_db")
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
    try:
        if client is not None:
            # Motor client's close() is synchronous
            client.close()
            print("MongoDB connection closed")
        else:
            print("No active MongoDB connection to close")
    except Exception as e:
        print(f"Error closing MongoDB connection: {e}")
