# LEGION

Multi-agent system for InfinitePay customer service using Gemini API, FastAPI, and RAG (Retrieval Augmented Generation). A modular, registry-based architecture that enables easy addition of new agents without modifying core routing logic.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Registry-Centric Design](#registry-centric-design)
- [Quick Start](#quick-start)
- [Adding New Agents](#adding-new-agents)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Docker Deployment](#docker-deployment)

## Overview

Legion is a multi-agent system where specialized agents handle different types of customer service queries. The system is built around a **central Agent Registry** that enables complete modularity - agents register themselves on initialization, and the router dynamically discovers and routes to them based on intent classification.

### Key Features

- **Registry-Based Architecture**: All agents auto-register on instantiation
- **Dynamic Intent Classification**: Router queries registry for available intents
- **Modular Design**: Add new agents without touching router code
- **RAG-Powered Knowledge**: Vector search over InfinitePay documentation
- **Tool-Based Support**: LangChain tool calling for account operations
- **Fully Dockerized**: Ready for containerized deployment

## Architecture

### System Flow

```
User Request
    ↓
FastAPI Endpoint
    ↓
Router Agent
    ├─ Classifies Intent (using registry)
    ├─ Discovers Agents (from registry)
    ├─ Selects Best Agent(s)
    └─ Executes Agent(s) in Parallel
        ↓
Agent(s) Process Request
    ├─ Knowledge Agent → RAG Retrieval → ChromaDB
    ├─ Support Agent → Tool Execution → SQLite DB
    └─ [Future Agents] → [Their Logic]
        ↓
Router Combines Responses (if multiple)
        ↓
Response to User
```

### Core Components

1. **Agent Registry** (`app/core/agent_registry.py`)
   - Central singleton that holds all agent metadata
   - Enables dynamic agent discovery
   - Provides intent-based and capability-based lookup

2. **Router Agent** (`app/agents/router.py`)
   - Orchestrates the entire system
   - Classifies user intent dynamically based on registered agents
   - Routes to appropriate agents without hardcoding

3. **Specialized Agents**
   - **Knowledge Agent**: RAG-based product information
   - **Support Agent**: Tool-based customer support
   - **Testing Agent**: (Future) Validation and testing

4. **Data Layer**
   - **ChromaDB**: Vector store for RAG (embedded, file-based)
   - **SQLite**: User database (file-based)

## Registry-Centric Design

### The Registry Pattern

The **Agent Registry** is the heart of the system. It enables complete modularity through a simple pattern:

1. **Auto-Registration**: Agents register themselves when instantiated
2. **Dynamic Discovery**: Router queries registry instead of hardcoding
3. **Intent-Based Routing**: Router discovers which agents handle which intents
4. **Zero Configuration**: No need to modify router when adding agents

### How It Works

#### 1. Agent Registration

When an agent is created, it automatically registers itself:

```python
class SupportAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="support_agent",
            description="Handles account and technical support",
            intents=[IntentType.CUSTOMER_SUPPORT],  # What it handles
            capabilities=["account_status", "transactions"],
            priority=5,  # Higher = preferred
            requires_user_id=True
        )
        # Registration happens automatically in BaseAgent.__init__()
```

The `BaseAgent` class automatically calls `AgentRegistry().register(self.metadata)` during initialization.

#### 2. Dynamic Intent Classification

The router doesn't hardcode intent categories. Instead, it queries the registry:

```python
# Router queries: "What intents have registered agents?"
available_intents = self.registry.get_available_intents()

# Builds classification prompt dynamically:
# "Available intents: product_info (handled by knowledge_agent), 
#                      customer_support (handled by support_agent)"
```

This means:
- **No hardcoding**: Intent list comes from registry
- **Self-documenting**: Prompt shows which agents handle what
- **Always up-to-date**: New agents automatically appear

#### 3. Agent Discovery

When routing, the router discovers agents dynamically:

```python
# Find all agents that handle this intent
candidates = self.registry.find_agents_by_intent(intent)

# Select best agent(s) based on priority
selected_agents = self._select_agents(candidates, context)

# Execute agents (parallel if multiple)
responses = await self._execute_agents(selected_agents, context)
```

#### 4. Benefits

- **Modularity**: Add agents without touching router
- **Flexibility**: Multiple agents can handle same intent
- **Priority System**: Handle ambiguous cases
- **Self-Documenting**: Registry knows what agents exist
- **Testable**: Easy to mock registry in tests

### Registry API

The `AgentRegistry` provides these methods:

```python
# Find agents by intent
agents = registry.find_agents_by_intent(IntentType.PRODUCT_INFO)

# Find agents by capability
agents = registry.find_agents_by_capability("rag_retrieval")

# Get all registered agents
all_agents = registry.get_all_agents()

# Get available intents (grouped by intent)
intents = registry.get_available_intents()

# Select best agent for intent
best = registry.select_best_agent(IntentType.CUSTOMER_SUPPORT)
```

## Adding New Agents

Adding a new agent is straightforward - no router modifications needed.

### Step 1: Create Agent Class

```python
# app/agents/billing_agent.py
from app.agents.base import BaseAgent
from app.models.agent_metadata import IntentType

class BillingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="billing_agent",
            description="Handles billing and payment questions",
            intents=[IntentType.PRODUCT_INFO, IntentType.CUSTOMER_SUPPORT],
            capabilities=["billing_info", "payment_processing"],
            priority=4,
            requires_user_id=False
        )
        # Your agent initialization here
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Your agent logic here
        return {
            "response": "Agent response",
            "agent": self.name,
            "metadata": {}
        }
```

### Step 2: Register on Startup

```python
# main.py
from app.agents.billing import BillingAgent

@app.on_event("startup")
async def startup():
    await db.initialize()
    support_agent = SupportAgent()
    knowledge_agent = KnowledgeAgent()
    billing_agent = BillingAgent()  # Just instantiate - auto-registers!
```

### Step 3: That's It!

The router will automatically:
- Discover the new agent
- Include its intents in classification
- Route to it when appropriate
- No code changes needed!

### Example: Complete New Agent

```python
from app.agents.base import BaseAgent
from app.models.agent_metadata import IntentType
from typing import Dict, Any

class FAQAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="faq_agent",
            description="Answers frequently asked questions",
            intents=[IntentType.GENERAL_QUESTION],
            capabilities=["faq_retrieval"],
            priority=2
        )
        self.faq_db = {}  # Your FAQ data source
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context["message"]
        # Your FAQ logic
        answer = self._search_faq(query)
        return {
            "response": answer,
            "agent": self.name,
            "metadata": {"source": "faq_db"}
        }
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- `.env` file with `GEMINI_API_KEY`

### Docker (Recommended)

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Local Development

```bash
# Setup
cd src/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../../requirements.txt

# Initialize data
python scripts/seed_mock_data.py
python scripts/ingest_data.py

# Run server
uvicorn main:app --reload
```

### Environment Variables

Create `.env` in project root:

```env
GEMINI_API_KEY=your_api_key_here
```

## API Documentation

### POST /api/chat

Main chat endpoint that routes through the agent system.

**Request:**
```json
{
  "message": "What are the fees for Maquininha Smart?",
  "user_id": "user_001"
}
```

**Response:**
```json
{
  "response": "The Maquininha Smart has a fee of 2.9% per transaction...",
  "agent": "knowledge_agent",
  "metadata": {
    "sources": ["https://www.infinitepay.io/maquininha"],
    "chunks_retrieved": 3,
    "confidence": 0.9,
    "processing_time_ms": 1234
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /api/agents

Returns all registered agents from the registry.

**Response:**
```json
[
  {
    "name": "knowledge_agent",
    "description": "Answers InfinitePay product questions using RAG",
    "capabilities": ["rag_retrieval", "product_info"],
    "intents": ["product_info", "general_question"],
    "priority": 3,
    "requires_user_id": false
  },
  ...
]
```

### POST /api/users

Create a new user.

**Request:**
```json
{
  "user_id": "user_004",
  "name": "Ana Costa",
  "email": "ana@example.com",
  "balance": 0.0,
  "status": "active"
}
```

### GET /api/health

System health check including agent status.

## Development

### Project Structure

```
src/backend/
├── app/
│   ├── agents/
│   │   ├── base.py              # BaseAgent with auto-registration
│   │   ├── router.py            # Router Agent (uses registry)
│   │   ├── knowledge.py         # Knowledge Agent (RAG)
│   │   └── support.py           # Support Agent (tools)
│   ├── core/
│   │   └── agent_registry.py    # Central registry (singleton)
│   ├── models/
│   │   └── agent_metadata.py    # Agent metadata models
│   ├── rag/
│   │   ├── embeddings.py       # Embedding service
│   │   ├── vectorstore.py      # ChromaDB wrapper
│   │   ├── retriever.py         # RAG retrieval
│   │   └── ingestion.py         # Data ingestion
│   ├── data/
│   │   └── database.py          # SQLite wrapper
│   ├── tools/
│   │   └── support_tools.py    # LangChain tools
│   └── utils/
│       └── logger.py            # Logging setup
├── routers/
│   ├── agent_router.py         # Agent API endpoints
│   └── user_router.py          # User management endpoints
├── scripts/
│   ├── ingest_data.py          # Build vector DB
│   └── seed_mock_data.py       # Seed user data
├── main.py                     # FastAPI app
├── config.py                   # Configuration
└── models.py                    # API models
```

### Data Ingestion

Build the vector database from InfinitePay documentation:

```bash
python scripts/ingest_data.py

# Resume if interrupted
python scripts/ingest_data.py --resume
```

This scrapes 18 InfinitePay URLs, chunks the content, generates embeddings, and stores in ChromaDB.

### Adding New Intent Types

1. Add to `IntentType` enum in `app/models/agent_metadata.py`:

```python
class IntentType(str, Enum):
    PRODUCT_INFO = "product_info"
    CUSTOMER_SUPPORT = "customer_support"
    BILLING_QUESTION = "billing_question"  # New intent
    ...
```

2. Create agent that handles it (see [Adding New Agents](#adding-new-agents))
3. Router automatically discovers it!

## Docker Deployment

### Dockerfile

Multi-stage build with Python 3.11, optimized for production.

### Docker Compose

Includes:
- Volume mounts for data persistence
- Environment variable injection
- Health checks
- Auto-restart on failure

### Data Persistence

All data persists in `./data/`:
- `data/users.db` - SQLite user database
- `data/vector_db/` - ChromaDB vector store

These are mounted as volumes, so data survives container restarts.

## Architecture Decisions

### Why Registry Pattern?

1. **Modularity**: Agents are independent modules
2. **Discoverability**: System knows what agents exist
3. **Flexibility**: Easy to add/remove agents
4. **Testability**: Mock registry for testing
5. **Scalability**: Add agents without touching core

### Why Dynamic Intent Classification?

- Router adapts to available agents automatically
- No hardcoding of intent categories
- Self-documenting system
- New intents appear automatically

### Why File-Based Databases?

- **SQLite**: Simple, reliable, no separate server
- **ChromaDB**: Embedded mode, perfect for Docker
- **Persistence**: Data survives restarts via volumes
- **Portability**: Easy to backup/restore

### Why Parallel Agent Execution?

- Multiple agents can handle same intent
- Faster response times
- Better user experience
- Router combines responses intelligently

## Tech Stack

- **Framework**: FastAPI 0.115.6
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Gemini embedding-001
- **Agent Framework**: LangChain 0.3.7
- **Vector DB**: ChromaDB 0.5.20 (embedded)
- **Database**: SQLite (via aiosqlite)
- **Web Scraping**: BeautifulSoup4, html2text
- **Deployment**: Docker, Docker Compose

## License

[Your License Here]
