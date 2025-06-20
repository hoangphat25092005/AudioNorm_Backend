from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from app.models.user import UserResponse, UserRegister, UserLogin, GoogleUser
from app.services.auth_service import AuthService
from app.config.oauth import oauth

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

@router.get("/google/login")
async def google_login(request: Request):
    """
    Initialize Google OAuth2 login flow
    """
    redirect_uri = request.url_for('google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/google/callback")
async def google_callback(request: Request):
    """
    Handle Google OAuth2 callback
    """
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')
    
    if user_info:
        google_user = GoogleUser(
            id=user_info['sub'],
            email=user_info['email'],
            verified_email=user_info['email_verified'],
            name=user_info.get('name', user_info['email'].split('@')[0]),
            picture=user_info.get('picture')
        )
        
        # Login or register the Google user
        result = await AuthService.google_auth(google_user)
        return result
    
    raise HTTPException(status_code=400, detail="Failed to get user info from Google")