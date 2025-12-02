import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.data.database import db


async def seed_mock_data():
    """Seed database with mock user data"""
    await db.initialize()
    
    mock_users = [
        {
            "user_id": "user_001",
            "name": "João Silva",
            "email": "joao.silva@example.com",
            "balance": 1250.50,
            "status": "active"
        },
        {
            "user_id": "user_002",
            "name": "Maria Santos",
            "email": "maria.santos@example.com",
            "balance": 3500.00,
            "status": "active"
        },
        {
            "user_id": "user_003",
            "name": "Pedro Oliveira",
            "email": "pedro.oliveira@example.com",
            "balance": 500.25,
            "status": "active"
        }
    ]
    
    mock_transactions = [
        {
            "transaction_id": "txn_001",
            "user_id": "user_001",
            "amount": 100.00,
            "type": "payment_received",
            "description": "Payment from customer"
        },
        {
            "transaction_id": "txn_002",
            "user_id": "user_001",
            "amount": -50.00,
            "type": "transfer",
            "description": "Transfer to account"
        },
        {
            "transaction_id": "txn_003",
            "user_id": "user_002",
            "amount": 250.00,
            "type": "payment_received",
            "description": "Payment from customer"
        }
    ]
    
    import aiosqlite
    
    async with aiosqlite.connect(db._db_path) as conn:
        for user in mock_users:
            await conn.execute(
                """INSERT OR REPLACE INTO users 
                   (user_id, name, email, balance, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (user["user_id"], user["name"], user["email"], 
                 user["balance"], user["status"])
            )
        
        for txn in mock_transactions:
            await conn.execute(
                """INSERT OR REPLACE INTO transactions 
                   (transaction_id, user_id, amount, type, description)
                   VALUES (?, ?, ?, ?, ?)""",
                (txn["transaction_id"], txn["user_id"], txn["amount"],
                 txn["type"], txn["description"])
            )
        
        await conn.commit()
    
    print(f"Seeded {len(mock_users)} users and {len(mock_transactions)} transactions")
    print(f"Database location: {db._db_path}")


if __name__ == "__main__":
    asyncio.run(seed_mock_data())

