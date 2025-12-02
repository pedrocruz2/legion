import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings


class VectorStore:
    _instance = None
    _client = None
    _collection = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            import os
            if os.getenv('DOCKER_ENV') or os.path.exists('/app/data'):
                data_dir = Path('/app/data') / "vector_db"
            else:
                current_file = Path(__file__).resolve()
                backend_dir = current_file.parent.parent.parent
                project_root = backend_dir.parent.parent
                data_dir = project_root / "data" / "vector_db"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            self._db_path = str(data_dir)
            self._collection_name = "scraped_docs"
            self._initialized = True
    
    def _get_client(self):
        """Get or create ChromaDB client"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self._db_path,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client
    
    def get_collection(self):
        """Get or create collection"""
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_collection(name=self._collection_name)
            except:
                self._collection = client.create_collection(
                    name=self._collection_name,
                    metadata={"description": "Scraped documentation chunks"}
                )
        return self._collection
    
    async def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """Add documents to the vector store"""
        collection = self.get_collection()
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        collection = self.get_collection()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results.get('distances') else 0.0
                similarity = 1.0 - distance
                
                if similarity >= min_score:
                    documents.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                        "similarity": similarity,
                        "id": results['ids'][0][i] if results.get('ids') else None
                    })
        
        return documents
    
    def clear_collection(self):
        """Clear all documents from collection (for re-ingestion)"""
        client = self._get_client()
        try:
            client.delete_collection(name=self._collection_name)
            self._collection = None
        except:
            pass
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        collection = self.get_collection()
        return collection.count()

