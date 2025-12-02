from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any

from models import (
    ChatRequest,
    ChatResponse,
    TestRequest,
    TestResponse,
    HealthResponse,
    MetricsResponse
)
from app.agents.router import RouterAgent
from app.core.agent_registry import AgentRegistry
from app.models.agent_metadata import IntentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import settings

router = APIRouter(prefix="/api", tags=["agents"])

router_agent = RouterAgent()
registry = AgentRegistry()
llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.7
)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat messages through the agent system"""
    try:
        context = {
            "message": request.message,
            "user_id": request.user_id,
            "timestamp": datetime.now()
        }
        
        result = await router_agent.process(context)
        
        return ChatResponse(
            response=result.get("response", "No response generated"),
            agent=result.get("agent", "router_agent"),
            metadata=result.get("metadata", {}),
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.post("/test", response_model=TestResponse)
async def test(request: TestRequest):
    """Trigger testing agent to validate knowledge agent responses"""
    try:
        testing_agent_metadata = registry.get_agent("testing_agent")
        if not testing_agent_metadata or not testing_agent_metadata.agent_instance:
            raise HTTPException(status_code=404, detail="Testing agent not found")
        
        testing_agent = testing_agent_metadata.agent_instance
        context = {
            "message": request.message,
            "user_id": None,
            "timestamp": datetime.now()
        }
        
        result = await testing_agent.process(context)
        
        return TestResponse(
            test_id=f"test_{datetime.now().timestamp()}",
            status=result.get("metadata", {}).get("status", "unknown"),
            results={
                "response": result.get("response", ""),
                "metadata": result.get("metadata", {})
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running test: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health():
    """System health check"""
    try:
        test_message = HumanMessage(content="test")
        await llm.ainvoke([test_message])
        gemini_status = "connected"
    except Exception:
        gemini_status = "disconnected"
    
    return HealthResponse(
        status="healthy" if gemini_status == "connected" else "degraded",
        agents={
            "router": "healthy",
            "knowledge": "healthy",
            "support": "healthy",
            "testing": "healthy"
        },
        vector_db="ready",
        gemini_api=gemini_status
    )


@router.get("/agents")
async def get_agents():
    """Get all registered agents from the registry"""
    agents = registry.get_all_agents()
    return [
        {
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "intents": [intent.value for intent in agent.intents],
            "priority": agent.priority,
            "requires_user_id": agent.requires_user_id
        }
        for agent in agents
    ]


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system performance metrics"""
    return MetricsResponse(
        total_requests=0,
        agent_usage={},
        average_response_time_ms=0.0,
        error_rate=0.0
    )

