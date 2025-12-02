from typing import Dict, Any, List, Tuple
import sys
import asyncio
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.agents.base import BaseAgent
from app.core.agent_registry import AgentRegistry
from app.models.agent_metadata import IntentType
from app.utils.logger import setup_logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = setup_logger("router_agent")


class RouterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="router_agent",
            description="Routes requests to appropriate agents",
            intents=[],
            capabilities=["routing", "intent_classification"]
        )
        self.registry = AgentRegistry()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.ROUTER_TEMPERATURE
        )
    
    async def _classify_intent(self, message: str) -> Tuple[IntentType, bool]:
        """Classify user message intent and determine if agent is needed.
        Dynamically builds prompt based on registered agents in the registry.
        Returns (intent, needs_agent)"""
        
        available_intents = self.registry.get_available_intents()
        
        if not available_intents:
            return IntentType.GENERAL_QUESTION, False
        
        intent_categories = []
        for intent, agents in available_intents.items():
            agent_names = [agent.name for agent in agents]
            intent_categories.append(
                f"- {intent.value}: Handled by {', '.join(agent_names)}"
            )
        
        intent_list = "\n".join(intent_categories)
        available_intent_values = [intent.value for intent in available_intents.keys()]
        
        prompt = f"""Analyze the following user message and determine:
        1. The intent category (must be one of: {', '.join(available_intent_values)})
        2. Whether a specialized agent is needed (true/false)

        Available intent categories (based on registered agents):
        {intent_list}

        Special cases:
        - casual_greeting: Simple greetings, casual conversation (eae, oi, hello, etc.) - set needs_agent to false
        - If the message doesn't match any intent above, use: general_question

        Message: "{message}"

        Respond in this exact format:
        intent: <intent_name>
        needs_agent: <true or false>

        If it's just a greeting or casual conversation, set needs_agent to false."""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip().lower()
            
            intent_str = None
            needs_agent = True
            
            for line in response_text.split('\n'):
                if 'intent:' in line:
                    intent_str = line.split('intent:')[1].strip()
                elif 'needs_agent:' in line:
                    needs_agent_str = line.split('needs_agent:')[1].strip()
                    needs_agent = needs_agent_str == 'true'
            
            dynamic_intent_map = {intent.value: intent for intent in available_intents.keys()}
            dynamic_intent_map["casual_greeting"] = IntentType.GENERAL_QUESTION
            
            intent = dynamic_intent_map.get(intent_str, IntentType.GENERAL_QUESTION)
            
            logger.info(
                f"Intent Classification - Message: '{message[:100]}...' | "
                f"Intent: {intent.value} | Needs Agent: {needs_agent}"
            )
            
            return intent, needs_agent
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            return IntentType.GENERAL_QUESTION, True
    
    def _select_agents(
        self, 
        candidates: List[Any], 
        context: Dict[str, Any]
    ) -> List[Any]:
        """Select which agent(s) to use from candidates"""
        if not candidates:
            return []
        
        if len(candidates) == 1:
            return [candidates[0].agent_instance]
        
        sorted_candidates = sorted(
            candidates, 
            key=lambda a: a.priority, 
            reverse=True
        )
        
        primary = sorted_candidates[0]
        selected = [primary.agent_instance]
        
        if len(sorted_candidates) > 1:
            secondary = sorted_candidates[1]
            if secondary.priority == primary.priority:
                selected.append(secondary.agent_instance)
        
        return selected
    
    async def _handle_direct_response(
        self, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle messages that don't need agents (greetings, casual chat)"""
        prompt = f"""You are a friendly customer service assistant. 
        Always respond in the language used by the user.

        User message: "{context['message']}"

        Provide a natural, friendly response:"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return {
                "response": response.content,
                "agent": self.name,
                "metadata": {
                    "confidence": 0.8,
                    "processing_time_ms": 0,
                    "direct_response": True
                }
            }
        except Exception as e:
            return {
                "response": "Hello! How can I help you today? / Olá! Como posso ajudar você hoje?",
                "agent": self.name,
                "metadata": {
                    "confidence": 0.5,
                    "processing_time_ms": 0,
                    "direct_response": True,
                    "error": str(e)
                }
            }
    
    async def _execute_agents(
        self,
        agents: List[Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute multiple agents in parallel and return their responses"""
        agent_names = [agent.name for agent in agents]
        logger.info(f"Executing agents: {agent_names} for message: '{context['message'][:100]}...'")
        
        tasks = [agent.process(context) for agent in agents]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Agent {agents[i].name} error: {str(response)}")
                results.append({
                    "response": f"Error from {agents[i].name}: {str(response)}",
                    "agent": agents[i].name,
                    "metadata": {"error": True}
                })
            else:
                agent_response = response.get("response", "")
                logger.info(
                    f"Agent {agents[i].name} Response - "
                    f"Length: {len(agent_response)} chars | "
                    f"Response: '{agent_response[:200]}...'"
                )
                results.append(response)
        
        return results
    
    async def _combine_responses(
        self,
        responses: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """Combine multiple agent responses into a single coherent response"""
        if len(responses) == 1:
            return responses[0].get("response", "No response generated")
        
        agent_responses = "\n\n".join([
            f"Agent {r.get('agent', 'unknown')}: {r.get('response', 'No response')}"
            for r in responses
        ])
        
        prompt = f"""You are a customer service assistant. 
        Combine the following responses from different agents into a single, coherent answer.
        Always respond in the language used by the user.

        Agent responses:
        {agent_responses}

        User question: "{context['message']}"

        Provide a unified, natural response that combines the information:"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception:
            return " / ".join([r.get("response", "") for r in responses])
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method - routes to appropriate agent(s)"""
        from datetime import datetime
        start_time = datetime.now()
        
        try:
            intent, needs_agent = await self._classify_intent(context["message"])
            
            if not needs_agent:
                result = await self._handle_direct_response(context)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                result["metadata"]["processing_time_ms"] = int(processing_time)
                
                logger.info(
                    f"Direct Response - Message: '{context['message'][:100]}...' | "
                    f"Response: '{result['response'][:100]}...' | "
                    f"Processing Time: {int(processing_time)}ms"
                )
                
                return result
            
            candidates = self.registry.find_agents_by_intent(intent)
            
            if not candidates:
                logger.warning(
                    f"No agents found for intent: {intent.value} | "
                    f"Message: '{context['message'][:100]}...' | "
                    f"Falling back to direct response"
                )
                result = await self._handle_direct_response(context)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                result["metadata"]["processing_time_ms"] = int(processing_time)
                result["metadata"]["note"] = "No specialized agent available, using direct response"
                return result
            
            selected_agents = self._select_agents(candidates, context)
            
            if not selected_agents:
                logger.warning(
                    f"No agents selected from candidates | "
                    f"Intent: {intent.value} | "
                    f"Falling back to direct response"
                )
                result = await self._handle_direct_response(context)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                result["metadata"]["processing_time_ms"] = int(processing_time)
                return result
            
            agent_responses = await self._execute_agents(selected_agents, context)
            combined_response = await self._combine_responses(agent_responses, context)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(
                f"Router Response - Intent: {intent.value} | "
                f"Selected Agents: {[agent.name for agent in selected_agents]} | "
                f"Response Length: {len(combined_response)} chars | "
                f"Processing Time: {int(processing_time)}ms"
            )
            
            return {
                "response": combined_response,
                "agent": self.name,
                "metadata": {
                    "intent": intent.value,
                    "selected_agents": [agent.name for agent in selected_agents],
                    "agent_responses": [r.get("agent") for r in agent_responses],
                    "confidence": 0.9,
                    "processing_time_ms": int(processing_time)
                }
            }
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(
                f"Router Error - Message: '{context.get('message', '')[:100]}...' | "
                f"Error: {str(e)} | "
                f"Processing Time: {int(processing_time)}ms"
            )
            return {
                "response": f"Error processing request: {str(e)} / Erro ao processar solicitação: {str(e)}",
                "agent": self.name,
                "metadata": {
                    "confidence": 0.0,
                    "processing_time_ms": int(processing_time),
                    "error": str(e)
                }
            }

