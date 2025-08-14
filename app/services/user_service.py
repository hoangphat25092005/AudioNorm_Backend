from app.models.user import UserRegister, UserResponse
from app.config.database import get_db
from passlib.context import CryptContext
from bson import ObjectId

try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception as e:
    # Fallback for bcrypt issues
    print(f"Warning: bcrypt initialization issue: {e}")
    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

class UserService:
    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    async def create_user(user: UserRegister) -> UserResponse:
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
        created_user["_id"] = str(created_user["_id"])  # Convert ObjectId to string
        return UserResponse(**created_user)

    @staticmethod
    async def get_user_by_id(user_id: str) -> UserResponse:
        db = await get_db()
        try:
            # Validate ObjectId format
            object_id = ObjectId(user_id)
            user = await db.users.find_one({"_id": object_id})
            if user:
                user["_id"] = str(user["_id"])  # Convert ObjectId to string
                return UserResponse(**user)
            return None
        except Exception as e:
            # Invalid ObjectId format
            return None

    @staticmethod
    async def get_user_by_email(email: str) ->  UserResponse:
        db = await get_db()
        user = await db.users.find_one({"email": email})
        if user:
            user["_id"] = str(user["_id"])  # Convert ObjectId to string
            return UserResponse(**user)
        return None
