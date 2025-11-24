from fastapi import APIRouter, HTTPException
from typing import List, Optional

from models import UserCreateRequest, UserResponse
from app.data.database import db

router = APIRouter(prefix="/api/users", tags=["users"])


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(request: UserCreateRequest):
    """Create a new user"""
    try:
        user = await db.create_user(
            user_id=request.user_id,
            name=request.name,
            email=request.email,
            balance=request.balance,
            status=request.status
        )
        return UserResponse(**user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get user by ID"""
    user = await db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return UserResponse(**user)


@router.get("", response_model=List[UserResponse])
async def list_users():
    """List all users (for development/testing)"""
    users = await db.list_all_users()
    return [UserResponse(**user) for user in users]

