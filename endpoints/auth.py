from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import re

security = HTTPBearer()

SECRET_KEY = "django-insecure-%pip@l9cra!+(1_cg2gjo#1f-=a)28h8et=la)0nf62m5t%k)p"
ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="JWT has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid JWT token")

def validate_email(email: str):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(pattern, email):
        raise HTTPException(status_code=422, detail="Invalid email format")
    return email

# FastAPI endpoint using plain Request
from fastapi import APIRouter

router = APIRouter()

@router.post("/forgot-password")
async def forgot_password(request: Request):
    body = await request.json()
    email = body.get("email")

    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    validate_email(email)

    # Continue with your logic (e.g., send OTP)
    return {"message": "OTP sent to email"}
