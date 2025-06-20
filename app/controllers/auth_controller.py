from fastapi import APIRouter, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from app.models.user import UserResponse, UserRegister, UserLogin
from app.services.auth_service import AuthService

router = APIRouter()

@router.post("/register", response_model=UserResponse, status_code=201)
async def register_user(user_data: UserRegister):
    """
    Register a new user
    """
    return await AuthService.register(user_data)

@router.post("/login")
async def login_user(user_data: UserLogin):
    """
    Login and get access token
    """
    return await AuthService.login(user_data)

# Optional: OAuth2 compatible login endpoint
@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user_data = UserLogin(username=form_data.username, password=form_data.password)
    return await AuthService.login(user_data)