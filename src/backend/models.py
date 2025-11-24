from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    message: str
    user_id: str


class TestRequest(BaseModel):
    message: str
    target_agent: Optional[str] = "knowledge_agent"


class ChatResponse(BaseModel):
    response: str
    agent: str
    metadata: Dict[str, Any]
    timestamp: datetime


class TestResponse(BaseModel):
    test_id: str
    status: str
    results: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    agents: Dict[str, str]
    vector_db: str
    gemini_api: str


class AgentInfo(BaseModel):
    name: str
    description: str
    capabilities: List[str]


class MetricsResponse(BaseModel):
    total_requests: int
    agent_usage: Dict[str, int]
    average_response_time_ms: float
    error_rate: float


class UserCreateRequest(BaseModel):
    user_id: str
    name: str
    email: Optional[str] = None
    balance: float = 0.0
    status: str = "active"


class UserResponse(BaseModel):
    user_id: str
    name: str
    email: Optional[str]
    balance: float
    status: str
    created_at: Optional[str] = None

