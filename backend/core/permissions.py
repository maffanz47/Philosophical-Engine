from fastapi import Depends, HTTPException
from .auth import get_current_user
from ..models.user import User

async def require_user(current_user: User = Depends(get_current_user)):
    # Any logged-in user (user or admin) passes
    return current_user

async def require_admin(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user