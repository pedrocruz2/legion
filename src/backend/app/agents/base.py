from abc import ABC, abstractmethod
from typing import Dict, Any, List
from app.core.agent_registry import AgentRegistry
from app.models.agent_metadata import AgentMetadata, IntentType


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        intents: List[IntentType],
        capabilities: List[str],
        priority: int = 0,
        requires_user_id: bool = False
    ):
        self.name = name
        self.metadata = AgentMetadata(
            name=name,
            description=description,
            intents=intents,
            capabilities=capabilities,
            priority=priority,
            requires_user_id=requires_user_id,
            agent_instance=self
        )
        AgentRegistry().register(self.metadata)
    
    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the request and return response"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health"""
        return {"status": "healthy"}

