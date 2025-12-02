import asyncio
import requests
from bs4 import BeautifulSoup
import html2text
from typing import List, Dict, Any
from pathlib import Path
import sys
import hashlib
import time
import json

from app.rag.embeddings import EmbeddingService
from app.rag.vectorstore import VectorStore

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings


class IngestionService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vectorstore = VectorStore()
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        
        self.rate_limit_delay = 1.2
        self.embedding_batch_delay = 2.0
        self.max_retries = 3
        self.retry_delay = 5
    
    def _chunk_text(self, text: str, url: str) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "url": url,
                    "chunk_index": chunk_index,
                    "start": start,
                    "end": end
                })
                chunk_index += 1
            
            start = end - self.chunk_overlap
        
        return chunks
    
    async def _scrape_url(self, url: str) -> str:
        """Scrape and extract text from URL"""
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = self.html_converter.handle(str(soup))
            return text.strip()
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return ""
    
    async def _embed_with_retry(self, texts: List[str], url: str) -> List[List[float]]:
        """Generate embeddings with retry logic and rate limiting"""
        for attempt in range(self.max_retries):
            try:
                embeddings = []
                for i, text in enumerate(texts):
                    if i > 0:
                        await asyncio.sleep(self.rate_limit_delay)
                    
                    embedding = await self.embedding_service.embed_text(text)
                    embeddings.append(embedding)
                    
                    if (i + 1) % 10 == 0:
                        print(f"    Embedded {i + 1}/{len(texts)} chunks...")
                        await asyncio.sleep(self.embedding_batch_delay)
                
                return embeddings
            except Exception as e:
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"  Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  Error generating embeddings: {str(e)}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        raise
        raise Exception("Failed to generate embeddings after retries")
    
    async def ingest_url(self, url: str):
        """Ingest a single URL with rate limiting"""
        print(f"\nProcessing {url}...")
        
        text = await self._scrape_url(url)
        
        if not text:
            print(f"  No content extracted from {url}")
            return 0
        
        print(f"  Extracted {len(text)} characters")
        
        chunks = self._chunk_text(text, url)
        print(f"  Created {len(chunks)} chunks")
        
        if not chunks:
            return 0
        
        texts = [chunk["text"] for chunk in chunks]
        print(f"  Generating embeddings (this may take a while due to rate limits)...")
        
        try:
            embeddings = await self._embed_with_retry(texts, url)
        except Exception as e:
            print(f"  Failed to generate embeddings: {str(e)}")
            return 0
        
        metadatas = [
            {
                "url": chunk["url"],
                "chunk_index": chunk["chunk_index"],
                "source": url
            }
            for chunk in chunks
        ]
        
        ids = [
            hashlib.md5(f"{url}_{chunk['chunk_index']}".encode()).hexdigest()
            for chunk in chunks
        ]
        
        await self.vectorstore.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  Stored {len(chunks)} chunks in vector DB")
        return len(chunks)
    
    async def ingest_all(self, urls: List[str] = None, resume: bool = False):
        """Ingest all URLs from config with progress tracking"""
        if urls is None:
            urls = settings.SCRAPE_URLS
        
        print(f"Starting ingestion of {len(urls)} URLs...")
        print(f"Vector DB location: {self.vectorstore._db_path}")
        print(f"Rate limit: {self.rate_limit_delay}s between embeddings")
        print(f"Batch delay: {self.embedding_batch_delay}s every 10 embeddings\n")
        
        if not resume:
            self.vectorstore.clear_collection()
            print("  Cleared existing collection")
        
        progress_file = Path(self.vectorstore._db_path).parent / "ingestion_progress.json"
        processed_urls = set()
        
        if resume and progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    processed_urls = set(progress.get('processed_urls', []))
                print(f"  Resuming from previous run ({len(processed_urls)} URLs already processed)")
            except:
                pass
        
        total_chunks = 0
        for i, url in enumerate(urls, 1):
            if url in processed_urls:
                print(f"\nSkipping {url} (already processed)")
                continue
            
            try:
                chunks = await self.ingest_url(url)
                total_chunks += chunks
                processed_urls.add(url)
                
                with open(progress_file, 'w') as f:
                    json.dump({'processed_urls': list(processed_urls)}, f)
                
                if i < len(urls):
                    delay = 3.0
                    print(f"  Waiting {delay}s before next URL (rate limit protection)...")
                    await asyncio.sleep(delay)
            except Exception as e:
                print(f"  Error processing {url}: {str(e)}")
                print(f"  You can resume later with --resume flag")
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    print(f"  Rate limit exceeded. Please wait and resume later.")
                    break
        
        if progress_file.exists():
            progress_file.unlink()
        
        print(f"\nIngestion complete!")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Collection size: {self.vectorstore.get_collection_count()}")

