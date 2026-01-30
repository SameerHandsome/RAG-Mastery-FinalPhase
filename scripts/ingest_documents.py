"""Script to ingest documents from a directory."""
import asyncio
from pathlib import Path
from loguru import logger
import sys
import argparse
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from app.services.llm_service import LLMService
from app.services.embeddings_service import EmbeddingsService
from app.services.cache_service import CacheService
from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.services.chunker_service import ChunkerService
from app.pipelines.ingestion_pipeline import IngestionPipeline


async def ingest_directory(directory: str):
    """Ingest all documents from a directory."""
    logger.info(f"Starting ingestion from: {directory}")
    
    try:
        # Initialize services
        cache_service = CacheService()
        llm_service = LLMService()
        embeddings_service = EmbeddingsService(cache_service)
        vector_store = VectorStore()
        graph_store = GraphStore(llm_service.llm)
        chunker_service = ChunkerService()
        
        ingestion_pipeline = IngestionPipeline(
            embeddings_service,
            vector_store,
            graph_store,
            chunker_service
        )
        
        # Find all supported files
        dir_path = Path(directory)
        supported_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        
        file_paths = []
        for ext in supported_extensions:
            file_paths.extend([str(p) for p in dir_path.glob(f"**/*{ext}")])
        
        if not file_paths:
            logger.warning(f"No supported files found in {directory}")
            return
        
        logger.info(f"Found {len(file_paths)} files to ingest")
        
        # Ingest
        result = await ingestion_pipeline.ingest(file_paths)
        
        logger.info(f"Ingestion completed:")
        logger.info(f"  Documents loaded: {result['documents_loaded']}")
        logger.info(f"  Chunks created: {result['chunks_created']}")
        logger.info(f"  Corpus version: {result['corpus_version']}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Graph RAG system")
    parser.add_argument("directory", help="Directory containing documents to ingest")
    
    args = parser.parse_args()
    
    asyncio.run(ingest_directory(args.directory))