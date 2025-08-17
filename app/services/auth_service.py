from app.models.user import (
    User, UserRegister, UserLogin, UserResponse,
    GoogleUser, AuthProvider
)
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
try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception as e:
    # Fallback for bcrypt issues
    print(f"Warning: bcrypt initialization issue: {e}")
    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

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
        """Register a new user and send verification email"""
        import secrets
        from app.services.email_service import EmailService

        db = await get_db()

        # Verify passwords match
        if user_data.password != user_data.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")

        # Check if username or email already exists
        if await db.users.find_one({"username": user_data.username}):
            raise HTTPException(status_code=400, detail="Username already registered")
        if await db.users.find_one({"email": user_data.email}):
            raise HTTPException(status_code=400, detail="Email already registered")

        # Generate verification token
        verification_token = secrets.token_urlsafe(32)

        # Create new user with is_verified=False and token
        user = User(
            username=user_data.username,
            email=user_data.email,
            password=AuthService.get_password_hash(user_data.password),
            created_at=datetime.utcnow(),
            is_verified=False,
            verification_token=verification_token
        )

        # Save to database
        result = await db.users.insert_one(user.dict())

        # Send verification email
        try:
            await EmailService.send_verification_email(user.email, user.username, verification_token)
        except Exception as e:
            print(f"Failed to send verification email: {e}")

        # Return user response (without password)
        return UserResponse(
            _id=str(result.inserted_id),
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
            data={
                "sub": user.username,
                "user_id": str(user_dict["_id"])  # MongoDB ObjectId as string
            }
        )

        # Return token and user data
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserResponse(
                _id=str(user_dict["_id"]),
                username=user.username,
                email=user.email,
                created_at=user.created_at,
                auth_provider=user.auth_provider
            )
        }

    @staticmethod
    async def google_auth(google_user: GoogleUser) -> dict:
        """Handle Google authentication"""
        db = await get_db()
        
        # Check if user exists
        user_dict = await db.users.find_one({"email": google_user.email})
        
        if not user_dict:
            # Create new user
            user = User(
                username=google_user.name,
                email=google_user.email,
                auth_provider=AuthProvider.GOOGLE,
                google_id=google_user.id,
                profile_picture=str(google_user.picture) if google_user.picture else None
            )
            result = await db.users.insert_one(user.dict())
            user_id = str(result.inserted_id)
        else:
            # Update existing user
            user = User(**user_dict)
            user_id = str(user_dict["_id"])
            if user.auth_provider != AuthProvider.GOOGLE:
                # Link Google account to existing user
                await db.users.update_one(
                    {"email": user.email},
                    {"$set": {
                        "auth_provider": AuthProvider.GOOGLE,
                        "google_id": google_user.id,
                        "profile_picture": str(google_user.picture) if google_user.picture else None
                    }}
                )

        # Create access token
        access_token = AuthService.create_access_token(
            data={
                "sub": google_user.email,
                "user_id": user_id
            }
        )

        # Return token and user data
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserResponse(
                _id=user_id,
                username=google_user.name,
                email=google_user.email,
                created_at=datetime.utcnow(),
                profile_picture=str(google_user.picture) if google_user.picture else None,
                auth_provider=AuthProvider.GOOGLE
            )
        }