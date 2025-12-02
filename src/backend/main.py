from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import agent_router, user_router
from app.data.database import db
from app.agents.support import SupportAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.testing import TestingAgent

app = FastAPI(
    title="Legion Backend",
    description="Multi-agent system for customer service with RAG-powered knowledge base",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize database and register agents on startup"""
    await db.initialize()
    support_agent = SupportAgent()
    knowledge_agent = KnowledgeAgent()
    testing_agent = TestingAgent()


@app.get("/")
async def root():
    return {"message": "Legion Backend API", "version": "1.0.0"}


app.include_router(agent_router.router)
app.include_router(user_router.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

