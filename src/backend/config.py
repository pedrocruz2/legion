from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


def find_env_file():
    """Find .env file in project root"""
    import os
    if os.getenv('DOCKER_ENV') or os.path.exists('/app/data'):
        env_file = Path('/app/.env')
        if env_file.exists():
            return str(env_file)
        return ".env"
    
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        return str(env_file)
    
    env_file_local = current_file.parent / ".env"
    if env_file_local.exists():
        return str(env_file_local)
    
    return ".env"


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/embedding-001"
    
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 5
    
    SCRAPE_URLS: List[str] = [
        "INSIRA_SUAS_ROTAS_AQUI",
        "https://example.com/page1",
        "https://example.com/page2"
    ]
    
    ENABLE_PASSIVE_TESTING: bool = True
    ROUTER_TEMPERATURE: float = 0.3
    KNOWLEDGE_TEMPERATURE: float = 0.7
    SUPPORT_TEMPERATURE: float = 0.5
    TESTING_TEMPERATURE: float = 0.4
    
    class Config:
        env_file = find_env_file()
        case_sensitive = True


settings = Settings()

