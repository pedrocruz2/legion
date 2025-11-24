import sys
from pathlib import Path
from typing import List
import google.generativeai as genai

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings


class EmbeddingService:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = settings.GEMINI_EMBEDDING_MODEL
    
    def _embed_sync(self, text: str, task_type: str) -> List[float]:
        """Synchronous embedding generation"""
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                raise Exception(f"QUOTA_EXCEEDED: {error_str}")
            raise Exception(f"Error generating embedding: {error_str}")
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        import asyncio
        return await asyncio.to_thread(self._embed_sync, text, "retrieval_document")
    
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query"""
        import asyncio
        return await asyncio.to_thread(self._embed_sync, text, "retrieval_query")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

