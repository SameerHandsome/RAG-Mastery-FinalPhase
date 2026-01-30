"""Setup script to initialize databases."""
import asyncio
from loguru import logger
import sys

sys.path.insert(0, '.')

from app.services.vector_store import VectorStore
from app.services.cache_service import CacheService


async def setup_databases():
    """Initialize all databases."""
    logger.info("Starting database setup...")
    
    try:
        # Setup Qdrant collection
        logger.info("Setting up Qdrant collection...")
        vector_store = VectorStore()
        await vector_store.setup_collection()
        logger.info("✓ Qdrant collection created")
        
        # Test Redis connection
        logger.info("Testing Redis connection...")
        cache_service = CacheService()
        await cache_service.set_query_response("test", "test")
        result = await cache_service.get_query_response("test")
        assert result == "test"
        logger.info("✓ Redis connection successful")
        
        logger.info("Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(setup_databases())