from typing import List, Dict, Any, Optional
from app.rag.embeddings import EmbeddingService
from app.rag.vectorstore import VectorStore
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from app.utils.logger import setup_logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = setup_logger("retriever")


class Retriever:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vectorstore = VectorStore()
        self.top_k = settings.RETRIEVAL_TOP_K
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.1
        )
    
    async def _translate_to_portuguese(self, query: str) -> str:
        """Translate query to Portuguese for better vector DB matching"""
        try:
            prompt = f"""Translate the following query to Portuguese (Brazil). 
If it's already in Portuguese, return it unchanged. 
Only return the translation, no explanations.

Query: "{query}"

Translation:"""
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            translated = response.content.strip().strip('"').strip("'")
            
            if translated != query:
                logger.info(f"Translated query to Portuguese - Original: '{query[:100]}...' | Translated: '{translated[:100]}...'")
            else:
                logger.debug(f"Query already in Portuguese: '{query[:100]}...'")
            
            return translated
        except Exception as e:
            logger.warning(f"Translation failed, using original query: {str(e)}")
            return query
    
    async def retrieve(self, query: str, top_k: Optional[int] = None, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query with retry logic.
        Translates query to Portuguese before embedding since all data is in Portuguese."""
        import asyncio
        
        if top_k is None:
            top_k = self.top_k
        
        translated_query = await self._translate_to_portuguese(query)
        
        for attempt in range(max_retries):
            try:
                query_embedding = await self.embedding_service.embed_query(translated_query)
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

