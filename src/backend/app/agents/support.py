from typing import Dict, Any
import sys
from pathlib import Path
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage

from app.agents.base import BaseAgent
from app.models.agent_metadata import IntentType
from app.tools.support_tools import (
    check_account_status,
    get_transaction_history,
    create_support_ticket,
    check_service_status
)
from app.utils.logger import setup_logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = setup_logger("support_agent")


class SupportAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="support_agent",
            description="Handles account and technical support issues",
            intents=[IntentType.CUSTOMER_SUPPORT],
            capabilities=[
                "account_status",
                "transaction_history",
                "support_tickets",
                "service_status"
            ],
            priority=5,
            requires_user_id=True
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.SUPPORT_TEMPERATURE
        )
        self.tools = [
            check_account_status,
            get_transaction_history,
            create_support_ticket,
            check_service_status
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any], user_id: str) -> str:
        """Execute a tool by name and return result as string"""
        tool_map = {
            "check_account_status": check_account_status,
            "get_transaction_history": get_transaction_history,
            "create_support_ticket": create_support_ticket,
            "check_service_status": check_service_status
        }
        
        tool = tool_map.get(tool_name)
        if not tool:
            return f"Error: Tool {tool_name} not found"
        
        try:
            if tool_name == "check_service_status":
                result = await tool.ainvoke({})
            elif tool_name == "create_support_ticket":
                result = await tool.ainvoke({"user_id": user_id, "issue": tool_args.get("issue", "")})
            elif tool_name == "get_transaction_history":
                result = await tool.ainvoke({
                    "user_id": user_id,
                    "limit": tool_args.get("limit", 10)
                })
            else:
                result = await tool.ainvoke({"user_id": user_id})
            
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process support requests using tools"""
        start_time = datetime.now()
        user_id = context.get("user_id", "")
        message = context.get("message", "")
        
        if not user_id:
            return {
                "response": "User ID is required for support requests. / ID do usuário é necessário para solicitações de suporte.",
                "agent": self.name,
                "metadata": {
                    "confidence": 0.0,
                    "error": "Missing user_id"
                }
            }
        
        system_prompt = f"""You are a helpful customer support agent.
        Always respond in the language used by the user.

        You have access to the following tools:
        - check_account_status: Check user account status and balance (requires user_id)
        - get_transaction_history: Get user's transaction history (requires user_id, optional limit)
        - create_support_ticket: Create a support ticket for issues (requires user_id and issue description)
        - check_service_status: Check service status (no parameters)

        Current user ID: {user_id}
        User message: "{message}"

        Use the appropriate tools to help the user. Be empathetic and provide clear next steps.
        After using tools, provide a natural response based on the tool results."""
        
        try:
            from langchain_core.messages import ToolMessage
            
            messages = [HumanMessage(content=system_prompt)]
            max_iterations = 5
            tools_used = []
            
            for iteration in range(max_iterations):
                response = await self.llm_with_tools.ainvoke(messages)
                messages.append(response)
                
                tool_calls = getattr(response, 'tool_calls', []) or []
                if not tool_calls:
                    break
                
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args", {})
                        tool_call_id = tool_call.get("id", "")
                    else:
                        tool_name = getattr(tool_call, "name", "")
                        tool_args = getattr(tool_call, "args", {})
                        tool_call_id = getattr(tool_call, "id", "")
                    
                    if not tool_name:
                        logger.warning(f"Tool call missing name, skipping: {tool_call}")
                        continue
                    
                    if not tool_call_id:
                        tool_call_id = f"call_{iteration}_{len(tools_used)}"
                    
                    tools_used.append(tool_name)
                    
                    try:
                        tool_result = await self._execute_tool(tool_name, tool_args, user_id)
                        
                        messages.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {str(e)}")
                        messages.append(
                            ToolMessage(
                                content=f"Error executing {tool_name}: {str(e)}",
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                        )
            
            final_response = messages[-1].content if messages else "I couldn't process your request."
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(
                f"Support Agent Response - User ID: {user_id} | "
                f"Message: '{message[:100]}...' | "
                f"Tools Used: {tools_used} | "
                f"Response: '{final_response[:200]}...' | "
                f"Processing Time: {int(processing_time)}ms"
            )
            
            return {
                "response": final_response,
                "agent": self.name,
                "metadata": {
                    "tools_used": tools_used,
                    "confidence": 0.9 if tools_used else 0.5,
                    "processing_time_ms": int(processing_time)
                }
            }
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(
                f"Support Agent Error - User ID: {user_id} | "
                f"Message: '{message[:100]}...' | "
                f"Error: {str(e)} | "
                f"Processing Time: {int(processing_time)}ms"
            )
            return {
                "response": f"I encountered an error while processing your request. Please try again. / Encontrei um erro ao processar sua solicitação. Por favor, tente novamente. Error: {str(e)}",
                "agent": self.name,
                "metadata": {
                    "confidence": 0.0,
                    "processing_time_ms": int(processing_time),
                    "error": str(e)
                }
            }

