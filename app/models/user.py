from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

# Like Mongoose Schema in Express
class User(BaseModel):
    username: str
    email: EmailStr
    password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

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

# For API responses (hide password)
class UserResponse(BaseModel):
    username: str
    email: EmailStr
    created_at: datetime