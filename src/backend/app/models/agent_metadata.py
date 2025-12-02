from pydantic import BaseModel
from typing import List, Optional, Any
from enum import Enum


class IntentType(str, Enum):
    PRODUCT_INFO = "product_info"
    CUSTOMER_SUPPORT = "customer_support"
    GENERAL_QUESTION = "general_question"
    SYSTEM_TESTING = "system_testing"


class AgentMetadata(BaseModel):
    name: str
    description: str
    intents: List[IntentType]
    capabilities: List[str]
    priority: int = 0
    requires_user_id: bool = False
    agent_instance: Optional[Any] = None

