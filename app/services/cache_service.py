from upstash_redis import Redis
from langsmith import traceable
from app.config import get_settings
from loguru import logger
import hashlib
import json


class CacheService:
    """Service for Upstash Redis caching using REST API."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize Upstash Redis client
        if self.settings.upstash_redis_rest_url and self.settings.upstash_redis_rest_token:
            self.redis = Redis(
                url=self.settings.upstash_redis_rest_url,
                token=self.settings.upstash_redis_rest_token
            )
            logger.info("Using Upstash Redis REST API")
        else:
            raise ValueError("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are required in .env")
    
    @traceable(name="cache_get_query")
    async def get_query_response(self, query: str) -> str | None:
        """Get cached query response."""
        try:
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            key = f"query:{query_hash}"
            
            cached = self.redis.get(key)
            if cached:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached
            
            logger.info(f"Cache miss for query: {query[:50]}...")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    @traceable(name="cache_set_query")
    async def set_query_response(self, query: str, response: str):
        """Cache query response with TTL."""
        try:
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            key = f"query:{query_hash}"
            
            ttl_seconds = self.settings.cache_ttl_hours * 3600
            self.redis.setex(key, ttl_seconds, response)
            
            logger.info(f"Cached response for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    @traceable(name="cache_get_embedding")
    async def get_embedding(self, text_hash: str) -> list[float] | None:
        """Get cached embedding."""
        try:
            key = f"embedding:{text_hash}"
            cached = self.redis.get(key)
            
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Embedding cache get error: {e}")
            return None
    
    @traceable(name="cache_set_embedding")
    async def set_embedding(self, text_hash: str, embedding: list[float]):
        """Cache embedding with TTL."""
        try:
            key = f"embedding:{text_hash}"
            ttl_seconds = self.settings.embedding_cache_ttl_hours * 3600
            
            self.redis.setex(key, ttl_seconds, json.dumps(embedding))
        except Exception as e:
            logger.error(f"Embedding cache set error: {e}")
    
    @traceable(name="cache_get_corpus")
    async def get_corpus(self, version: str) -> bytes | None:
        """Get cached BM25 corpus."""
        try:
            key = f"bm25_corpus:{version}"
            cached = self.redis.get(key)
            
            if cached:
                # Upstash returns string, convert to bytes
                if isinstance(cached, str):
                    import base64
                    return base64.b64decode(cached)
                return cached
            return None
        except Exception as e:
            logger.error(f"Corpus cache get error: {e}")
            return None
    
    @traceable(name="cache_set_corpus")
    async def set_corpus(self, version: str, corpus_data: bytes):
        """Cache BM25 corpus with TTL."""
        try:
            key = f"bm25_corpus:{version}"
            ttl_seconds = self.settings.corpus_cache_ttl_hours * 3600
            
            # Upstash REST API works with base64 for binary data
            import base64
            corpus_str = base64.b64encode(corpus_data).decode('utf-8')
            
            self.redis.setex(key, ttl_seconds, corpus_str)
            logger.info(f"Cached corpus version: {version}")
        except Exception as e:
            logger.error(f"Corpus cache set error: {e}")
    
    async def clear_all(self):
        """Clear all caches (for testing)."""
        try:
            # Get all keys and delete them
            # Note: Upstash has limitations on FLUSHDB in free tier
            logger.warning("Clearing all caches...")
            
            # Delete specific key patterns
            for pattern in ["query:*", "embedding:*", "bm25_corpus:*"]:
                keys = self.redis.keys(pattern)
                if keys:
                    for key in keys:
                        self.redis.delete(key)
            
            logger.info("Caches cleared")
        except Exception as e:
            logger.error(f"Clear cache error: {e}")
            logger.warning("Note: Upstash free tier has limited FLUSHDB access")