from typing import List, Dict, Any, Optional
from app.rag.embeddings import EmbeddingService
from app.rag.vectorstore import VectorStore
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings


class Retriever:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vectorstore = VectorStore()
        self.top_k = settings.RETRIEVAL_TOP_K
    
    async def retrieve(self, query: str, top_k: Optional[int] = None, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query with retry logic"""
        import asyncio
        
        if top_k is None:
            top_k = self.top_k
        
        for attempt in range(max_retries):
            try:
                query_embedding = await self.embedding_service.embed_query(query)
                results = await self.vectorstore.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    min_score=0.3
                )
                return results
            except Exception as e:
                error_str = str(e)
                if "QUOTA_EXCEEDED" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise Exception("Embedding quota exceeded. Please try again later.")
                raise
    
    async def retrieve_with_sources(self, query: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """Retrieve chunks and extract unique source URLs"""
        chunks = await self.retrieve(query)
        sources = list(set([
            chunk["metadata"].get("url", "")
            for chunk in chunks
            if chunk["metadata"].get("url")
        ]))
        return chunks, sources

