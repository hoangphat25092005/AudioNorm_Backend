from app.models.user import UserCreate, UserInDB
from app.config.database import get_db
from passlib.context import CryptContext
from bson import ObjectId

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserService:
    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    async def create_user(user: UserCreate) -> UserInDB:
        db = await get_db()
        
        # Check if user already exists
        if await db.users.find_one({"email": user.email}):
            raise ValueError("Email already registered")
        if await db.users.find_one({"username": user.username}):
            raise ValueError("Username already taken")
        
        # Create user document
        user_dict = user.dict(exclude={"confirm_password"})
        user_dict["hashed_password"] = UserService.get_password_hash(user.password)
        del user_dict["password"]
        
        # Insert into database
        result = await db.users.insert_one(user_dict)
        
        # Fetch and return created user
        created_user = await db.users.find_one({"_id": result.inserted_id})
        return UserInDB(**created_user)

    @staticmethod
    async def get_user_by_id(user_id: str) -> UserInDB:
        db = await get_db()
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if user:
            return UserInDB(**user)
        return None

    @staticmethod
    async def get_user_by_email(email: str) -> UserInDB:
        db = await get_db()
        user = await db.users.find_one({"email": email})
        if user:
            return UserInDB(**user)
        return None
