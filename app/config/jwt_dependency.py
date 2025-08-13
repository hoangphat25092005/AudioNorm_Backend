from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
#load_dotenv for Algorithm and SECRET_KEY
from dotenv import load_dotenv
import os

load_dotenv()
# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=401, 
                detail="Invalid token: no user ID"
            )
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")