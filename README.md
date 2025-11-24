# Agent Swarm Backend

Multi-agent system for InfinitePay customer service using Gemini API, FastAPI, and RAG.

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- `.env` file with `GEMINI_API_KEY` set

### Run with Docker Compose

```bash
docker-compose up --build
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Data Persistence
- Database: `./data/users.db` (persisted)
- Vector DB: `./data/vector_db/` (persisted)

## Development Setup

### Local Development

```bash
cd src/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r ../../requirements.txt
uvicorn main:app --reload
```

### Initial Setup

1. **Seed mock data:**
```bash
python scripts/seed_mock_data.py
```

2. **Ingest documentation (one-time):**
```bash
python scripts/ingest_data.py
```

## Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

## API Endpoints

- `POST /api/chat` - Chat with agents
- `GET /api/health` - Health check
- `GET /api/agents` - List all agents
- `POST /api/users` - Create user
- `GET /api/users/{user_id}` - Get user

## Architecture

- **Router Agent**: Classifies intent and routes to appropriate agents
- **Knowledge Agent**: RAG-based answers using ChromaDB
- **Support Agent**: Tool-based customer support

