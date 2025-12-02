from typing import Dict, Any
from langchain.tools import tool
from app.data.database import db


@tool
async def check_account_status(user_id: str) -> Dict[str, Any]:
    """Check user account status and balance.
    
    Args:
        user_id: The user's unique identifier
        
    Returns:
        Dictionary with account information including status, balance, and name
    """
    user = await db.get_user(user_id)
    if not user:
        return {
            "error": "User not found",
            "user_id": user_id
        }
    
    return {
        "user_id": user["user_id"],
        "name": user["name"],
        "status": user["status"],
        "balance": user["balance"],
        "email": user.get("email")
    }


@tool
async def get_transaction_history(user_id: str, limit: int = 10) -> Dict[str, Any]:
    """Get user's transaction history.
    
    Args:
        user_id: The user's unique identifier
        limit: Maximum number of transactions to return (default: 10)
        
    Returns:
        Dictionary with list of transactions
    """
    if not await db.user_exists(user_id):
        return {
            "error": "User not found",
            "user_id": user_id
        }
    
    transactions = await db.get_user_transactions(user_id, limit)
    return {
        "user_id": user_id,
        "transactions": transactions,
        "count": len(transactions)
    }


@tool
async def create_support_ticket(user_id: str, issue: str) -> Dict[str, Any]:
    """Create a support ticket for a user issue.
    
    Args:
        user_id: The user's unique identifier
        issue: Description of the issue
        
    Returns:
        Dictionary with ticket information
    """
    if not await db.user_exists(user_id):
        return {
            "error": "User not found",
            "user_id": user_id
        }
    
    ticket_id = await db.create_support_ticket(user_id, issue)
    return {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "issue": issue,
        "status": "open",
        "message": "Support ticket created successfully"
    }


@tool
async def check_service_status() -> Dict[str, Any]:
    """Check the status of services.
    
    Returns:
        Dictionary with service status information
    """
    return {
        "status": "operational",
        "services": {
            "api": "online",
            "payments": "online",
            "transactions": "online"
        },
        "last_updated": "2024-01-01T00:00:00Z"
    }

