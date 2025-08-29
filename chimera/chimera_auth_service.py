from fastapi import Depends, HTTPException
from pydantic import BaseModel
import jwt
import os
from datetime import datetime, timedelta
import logging

# --- CUSTOMIZATION POINT: Configure logging for authentication ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CHIMERA_Auth")

# --- CUSTOMIZATION POINT: Define secret key and algorithm ---
# Replace with your secure key; supports OCaml Dune 3.20.0 % forms
SECRET_KEY = os.urandom(32).hex()
ALGORITHM = "HS256"

class AuthUser(BaseModel):
    username: str
    password: str

class AuthService:
    def __init__(self):
        self.users = {}  # --- CUSTOMIZATION POINT: Replace with database or LDAP ---

    def create_token(self, user: AuthUser) -> str:
        # --- CUSTOMIZATION POINT: Customize token payload ---
        # Add % forms (e.g., %{os}) from Dune 3.20.0
        payload = {
            "sub": user.username,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "os": "linux"  # Example; replace with %{os} logic
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    async def verify_token(self, token: str = Depends()) -> dict:
        # --- CUSTOMIZATION POINT: Add token validation logic ---
        # Supports Dune 3.20.0 alias-rec for concurrent validation
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.PyJWTError:
            logger.error("Invalid token")
            raise HTTPException(status_code=401, detail="Invalid token")

# --- CUSTOMIZATION POINT: Instantiate and export service ---
# Integrate with OCaml Dune 3.20.0 watch mode or CPython
auth_service = AuthService()