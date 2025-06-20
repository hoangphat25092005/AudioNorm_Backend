from app.models.user import User, UserRegister, UserLogin, UserResponse
from app.config.database import get_db
from fastapi import HTTPException
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Password hashing settings
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")  # Use environment variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class AuthService:
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: dict) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    async def register(user_data: UserRegister) -> UserResponse:
        """Register a new user"""
        # Get database connection
        db = await get_db()
        
        # Verify passwords match
        if user_data.password != user_data.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")

        # Check if username or email already exists
        if await db.users.find_one({"username": user_data.username}):
            raise HTTPException(status_code=400, detail="Username already registered")
        if await db.users.find_one({"email": user_data.email}):
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create new user
        user = User(
            username=user_data.username,
            email=user_data.email,
            password=AuthService.get_password_hash(user_data.password),
            created_at=datetime.utcnow()
        )

        # Save to database
        await db.users.insert_one(user.dict())

        # Return user response (without password)
        return UserResponse(
            username=user.username,
            email=user.email,
            created_at=user.created_at
        )

    @staticmethod
    async def login(user_data: UserLogin) -> dict:
        """Login a user"""
        # Get database connection
        db = await get_db()
        
        # Find user
        user_dict = await db.users.find_one({"username": user_data.username})
        if not user_dict:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        # Convert to User model
        user = User(**user_dict)

        # Verify password
        if not AuthService.verify_password(user_data.password, user.password):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        # Create access token
        access_token = AuthService.create_access_token(
            data={"sub": user.username}
        )

        # Return token and user data
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserResponse(
                username=user.username,
                email=user.email,
                created_at=user.created_at
            )
        }