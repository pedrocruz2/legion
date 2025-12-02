from typing import Dict, Any
import sys
from pathlib import Path
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.agents.base import BaseAgent
from app.models.agent_metadata import IntentType
from app.rag.retriever import Retriever
from app.utils.logger import setup_logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = setup_logger("knowledge_agent")


class KnowledgeAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="knowledge_agent",
            description="Answers product questions using RAG",
            intents=[IntentType.PRODUCT_INFO, IntentType.GENERAL_QUESTION],
            capabilities=["rag_retrieval", "product_info", "web_search"],
            priority=3,
            requires_user_id=False
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.KNOWLEDGE_TEMPERATURE
        )
        self.retriever = Retriever()
    
    def _build_context_prompt(self, chunks: list, query: str) -> str:
        """Build prompt with retrieved context"""
        context_text = "\n\n---\n\n".join([
            f"Source: {chunk['metadata'].get('url', 'Unknown')}\n{chunk['text']}"
            for chunk in chunks
        ])
        
        sources = list(set([
            chunk['metadata'].get('url', '')
            for chunk in chunks
            if chunk['metadata'].get('url')
        ]))
        
        sources_text = "\n".join([f"- {url}" for url in sources])
        
        prompt = f"""You are an expert assistant that answers questions using provided documentation.
Always respond in the language used by the user.

Context from documentation:
{context_text}

User question: "{query}"

Instructions:
- Answer ONLY using the provided context
- If the context doesn't contain enough information, say so clearly
- Always cite your sources by mentioning the URL
- Be accurate and concise

Answer:"""
        
        return prompt, sources
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process knowledge requests using RAG"""
        start_time = datetime.now()
        message = context.get("message", "")
        
        try:
            logger.info(f"Knowledge Agent - Retrieving context for: '{message[:100]}...'")
            
            try:
                chunks, sources = await self.retriever.retrieve_with_sources(message)
            except Exception as e:
                error_str = str(e)
                if "quota" in error_str.lower() or "429" in error_str:
                    logger.error(f"Quota exceeded for embeddings: {error_str}")
                    return {
                        "response": "I'm currently unable to search our documentation due to API rate limits. Please try again in a few moments. / No momento, não consigo buscar em nossa documentação devido a limites de API. Por favor, tente novamente em alguns instantes.",
                        "agent": self.name,
                        "metadata": {
                            "sources": [],
                            "chunks_retrieved": 0,
                            "confidence": 0.0,
                            "processing_time_ms": 0,
                            "error": "quota_exceeded"
                        }
                    }
                raise
            
            if not chunks:
                logger.warning(f"No relevant chunks found for query: '{message[:100]}...'")
                return {
                    "response": "I don't have specific information about that in our documentation. Could you rephrase your question? / Não tenho informações específicas sobre isso em nossa documentação. Você poderia reformular sua pergunta?",
                    "agent": self.name,
                    "metadata": {
                        "sources": [],
                        "chunks_retrieved": 0,
                        "confidence": 0.0,
                        "processing_time_ms": 0
                    }
                }
            
            logger.info(f"Retrieved {len(chunks)} chunks with {len(sources)} unique sources")
            
            prompt, sources_list = self._build_context_prompt(chunks, message)
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            final_response = response.content
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(
                f"Knowledge Agent Response - Query: '{message[:100]}...' | "
                f"Chunks: {len(chunks)} | "
                f"Sources: {len(sources_list)} | "
                f"Response Length: {len(final_response)} chars | "
                f"Processing Time: {int(processing_time)}ms"
            )
            
            return {
                "response": final_response,
                "agent": self.name,
                "metadata": {
                    "sources": sources_list,
                    "chunks_retrieved": len(chunks),
                    "confidence": 0.9 if chunks else 0.3,
                    "processing_time_ms": int(processing_time)
                }
            }
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(
                f"Knowledge Agent Error - Query: '{message[:100]}...' | "
                f"Error: {str(e)} | "
                f"Processing Time: {int(processing_time)}ms"
            )
            return {
                "response": f"I encountered an error while searching our documentation. Please try again. / Encontrei um erro ao buscar em nossa documentação. Por favor, tente novamente. Error: {str(e)}",
                "agent": self.name,
                "metadata": {
                    "sources": [],
                    "chunks_retrieved": 0,
                    "confidence": 0.0,
                    "processing_time_ms": int(processing_time),
                    "error": str(e)
                }
            }

