from typing import Dict, List, Optional
from app.models.agent_metadata import AgentMetadata, IntentType


class AgentRegistry:
    _instance = None
    _agents: Dict[str, AgentMetadata] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, metadata: AgentMetadata):
        """Register an agent"""
        self._agents[metadata.name] = metadata
    
    def get_agent(self, name: str) -> Optional[AgentMetadata]:
        """Get agent by name"""
        return self._agents.get(name)
    
    def find_agents_by_intent(self, intent: IntentType) -> List[AgentMetadata]:
        """Find all agents that handle this intent"""
        return [
            agent for agent in self._agents.values()
            if intent in agent.intents
        ]
    
    def find_agents_by_capability(self, capability: str) -> List[AgentMetadata]:
        """Find agents with specific capability"""
        return [
            agent for agent in self._agents.values()
            if capability in agent.capabilities
        ]
    
    def get_all_agents(self) -> List[AgentMetadata]:
        """Get all registered agents"""
        return list(self._agents.values())
    
    def select_best_agent(self, intent: IntentType) -> Optional[AgentMetadata]:
        """Select best agent for intent (highest priority)"""
        candidates = self.find_agents_by_intent(intent)
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.priority)
    
    def get_available_intents(self) -> Dict[IntentType, List[AgentMetadata]]:
        """Get all intents that have registered agents, grouped by intent"""
        intent_map: Dict[IntentType, List[AgentMetadata]] = {}
        for agent in self._agents.values():
            for intent in agent.intents:
                if intent not in intent_map:
                    intent_map[intent] = []
                intent_map[intent].append(agent)
        return intent_map
    
    def get_intent_descriptions(self) -> Dict[str, str]:
        """Get descriptions of what each intent handles, based on registered agents"""
        intent_descriptions = {}
        intent_map = self.get_available_intents()
        
        for intent, agents in intent_map.items():
            descriptions = [agent.description for agent in agents]
            intent_descriptions[intent.value] = f"Handled by: {', '.join(set(descriptions))}"
        
        return intent_descriptions

