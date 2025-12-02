import aiosqlite
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings


class Database:
    _instance = None
    _db_path: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            import os
            if os.getenv('DOCKER_ENV') or os.path.exists('/app/data'):
                data_dir = Path('/app/data')
            else:
                current_file = Path(__file__).resolve()
                backend_dir = current_file.parent.parent.parent
                project_root = backend_dir.parent.parent
                data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            self._db_path = str(data_dir / "users.db")
            self._initialized = True
    
    async def initialize(self):
        """Create tables if they don't exist"""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    balance REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS support_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    issue TEXT NOT NULL,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            await db.commit()
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
                return None
    
    async def get_user_transactions(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get user transactions"""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM transactions 
                   WHERE user_id = ? 
                   ORDER BY created_at DESC 
                   LIMIT ?""",
                (user_id, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def create_support_ticket(
        self, 
        user_id: str, 
        issue: str
    ) -> str:
        """Create a support ticket"""
        import uuid
        ticket_id = str(uuid.uuid4())
        
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO support_tickets (ticket_id, user_id, issue)
                   VALUES (?, ?, ?)""",
                (ticket_id, user_id, issue)
            )
            await db.commit()
        
        return ticket_id
    
    async def user_exists(self, user_id: str) -> bool:
        """Check if user exists"""
        user = await self.get_user(user_id)
        return user is not None
    
    async def create_user(
        self,
        user_id: str,
        name: str,
        email: Optional[str] = None,
        balance: float = 0.0,
        status: str = "active"
    ) -> Dict[str, Any]:
        """Create a new user"""
        async with aiosqlite.connect(self._db_path) as db_conn:
            try:
                await db_conn.execute(
                    """INSERT INTO users (user_id, name, email, balance, status)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, name, email, balance, status)
                )
                await db_conn.commit()
                return await self.get_user(user_id)
            except aiosqlite.IntegrityError:
                raise ValueError(f"User with ID {user_id} already exists")
    
    async def list_all_users(self) -> List[Dict[str, Any]]:
        """Get all users"""
        async with aiosqlite.connect(self._db_path) as db_conn:
            db_conn.row_factory = aiosqlite.Row
            async with db_conn.execute("SELECT * FROM users") as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]


db = Database()

