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
    
    INFINITEPAY_URLS: List[str] = [
        "https://www.infinitepay.io",
        "https://www.infinitepay.io/maquininha",
        "https://www.infinitepay.io/maquininha-celular",
        "https://www.infinitepay.io/tap-to-pay",
        "https://www.infinitepay.io/pdv",
        "https://www.infinitepay.io/receba-na-hora",
        "https://www.infinitepay.io/gestao-de-cobranca",
        "https://www.infinitepay.io/gestao-de-cobranca-2",
        "https://www.infinitepay.io/link-de-pagamento",
        "https://www.infinitepay.io/loja-online",
        "https://www.infinitepay.io/boleto",
        "https://www.infinitepay.io/conta-digital",
        "https://www.infinitepay.io/conta-pj",
        "https://www.infinitepay.io/pix",
        "https://www.infinitepay.io/pix-parcelado",
        "https://www.infinitepay.io/emprestimo",
        "https://www.infinitepay.io/cartao",
        "https://www.infinitepay.io/rendimento"
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

