from pydantic import BaseModel, EmailStr, Field, HttpUrl
from typing import Optional
from datetime import datetime
from enum import Enum

class AuthProvider(str, Enum):
    LOCAL = "local"
    GOOGLE = "google"

# Like Mongoose Schema in Express
class User(BaseModel):
    username: str
    email: EmailStr
    password: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    auth_provider: AuthProvider = AuthProvider.LOCAL
    google_id: Optional[str] = None
    profile_picture: Optional[HttpUrl] = None

# For registration endpoint
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str

# For login endpoint
class UserLogin(BaseModel):
    username: str
    password: str

# For Google OAuth user info
class GoogleUser(BaseModel):
    id: str
    email: EmailStr
    verified_email: bool
    name: str
    picture: Optional[HttpUrl] = None

# For API responses (hide password)
class UserResponse(BaseModel):
    _id: Optional[str] = None
    username: str
    email: EmailStr
    created_at: datetime
    profile_picture: Optional[HttpUrl] = None
    auth_provider: AuthProvider = AuthProvider.LOCAL 