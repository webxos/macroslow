# legal_auth_service.py
# Description: Authentication service for the Lawmakers Suite 2048-AES, inspired by CHIMERA 2048. Implements JWT-based role-based access control for legal research tasks. Integrates with MAML workflows and Angular frontend to ensure secure access for law students and faculty.

from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_jwt_secret_key")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_jwt_token(data: dict, expires_delta: timedelta = timedelta(minutes=60)):
    """
    Create a JWT token for user authentication.
    Args:
        data (dict): User data (e.g., {"user_id": "student1", "role": "admin"}).
        expires_delta (timedelta): Token expiration time.
    Returns:
        str: JWT token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt_token(token: str):
    """
    Verify JWT token and extract user data.
    Args:
        token (str): JWT token.
    Returns:
        dict: Decoded user data or None if invalid.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

if __name__ == "__main__":
    token = create_jwt_token({"user_id": "student1", "role": "admin"})
    print(f"JWT Token: {token}")
    payload = verify_jwt_token(token)
    print(f"Verified Payload: {payload}")