import asyncio
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.rag.ingestion import IngestionService


async def main():
    """Run data ingestion"""
    parser = argparse.ArgumentParser(description='Ingest documentation from configured URLs')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    args = parser.parse_args()
    
    ingestion = IngestionService()
    await ingestion.ingest_all(resume=args.resume)


if __name__ == "__main__":
    asyncio.run(main())

