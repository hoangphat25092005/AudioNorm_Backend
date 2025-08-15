from pydantic import BaseModel, EmailStr, Field, HttpUrl, ConfigDict
from typing import Optional
from datetime import datetime
from enum import Enum

class AuthProvider(str, Enum):
    LOCAL = "local"
    GOOGLE = "google"

# Like Mongoose Schema in Express
class User(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "password": "hashedpassword",
                "created_at": "2025-08-14T12:00:00Z",
                "auth_provider": "local",
                "google_id": None,
                "profile_picture": None
            }
        }
    )
    
    username: str
    email: EmailStr
    password: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    auth_provider: AuthProvider = AuthProvider.LOCAL
    google_id: Optional[str] = None
    profile_picture: Optional[HttpUrl] = None

# For registration endpoint
class UserRegister(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "password": "securepassword123",
                "confirm_password": "securepassword123"
            }
        }
    )
    
    username: str
    email: EmailStr
    password: str
    confirm_password: str

# For login endpoint
class UserLogin(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "johndoe",
                "password": "securepassword123"
            }
        }
    )
    
    username: str
    password: str

# For Google OAuth user info
class GoogleUser(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "google_user_id_123",
                "email": "john@gmail.com",
                "verified_email": True,
                "name": "John Doe",
                "picture": "https://lh3.googleusercontent.com/a/picture"
            }
        }
    )
    
    id: str
    email: EmailStr
    verified_email: bool
    name: str
    picture: Optional[HttpUrl] = None

# For API responses (hide password)
class UserResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "created_at": "2025-08-14T12:00:00Z",
                "profile_picture": None,
                "auth_provider": "local"
            }
        }
    )
    
    username: str
    email: EmailStr
    created_at: datetime
    profile_picture: Optional[HttpUrl] = None
    auth_provider: AuthProvider = AuthProvider.LOCAL 