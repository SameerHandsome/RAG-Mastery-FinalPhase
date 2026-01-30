from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langsmith import traceable
from app.config import get_settings
from app.services.cache_service import CacheService
from loguru import logger
import hashlib
import pickle


class EmbeddingsService:
    """Service for dense and sparse embeddings with corpus versioning."""
    
    def __init__(self, cache_service: CacheService):
        self.settings = get_settings()
        self.cache = cache_service
        
        # Dense embeddings
        self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sparse embeddings (BM25)
        self.bm25 = None
        self.tokenized_corpus = None
    
    @traceable(name="embed_dense")
    async def embed_dense(self, texts: list[str], use_cache: bool = True) -> list[list[float]]:
        """Generate dense embeddings with caching."""
        embeddings = []
        
        for text in texts:
            # Check cache
            if use_cache:
                cache_key = hashlib.sha256(text.encode()).hexdigest()
                cached = await self.cache.get_embedding(cache_key)
                if cached:
                    embeddings.append(cached)
                    continue
            
            # Generate embedding
            embedding = self.dense_model.encode(text).tolist()
            
            # Cache if enabled
            if use_cache:
                await self.cache.set_embedding(cache_key, embedding)
            
            embeddings.append(embedding)
        
        return embeddings
    
    @traceable(name="embed_sparse")
    async def embed_sparse(self, texts: list[str]) -> list[dict]:
        """Generate sparse BM25 embeddings using loaded corpus."""
        if not self.bm25:
            raise ValueError("BM25 corpus not loaded. Call load_corpus() first.")
        
        sparse_embeddings = []
        for text in texts:
            tokenized = text.lower().split()
            scores = self.bm25.get_scores(tokenized)
            
            # Convert to sparse format (index: score)
            sparse_dict = {
                str(i): float(score) 
                for i, score in enumerate(scores) 
                if score > 0
            }
            sparse_embeddings.append(sparse_dict)
        
        return sparse_embeddings
    
    @traceable(name="save_corpus")
    async def save_corpus(self, texts: list[str]):
        """Save tokenized corpus to Redis with versioning."""
        try:
            # Tokenize corpus
            self.tokenized_corpus = [text.lower().split() for text in texts]
            
            # Initialize BM25
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            # Serialize and save to Redis
            corpus_data = pickle.dumps(self.tokenized_corpus)
            await self.cache.set_corpus(self.settings.bm25_corpus_version, corpus_data)
            
            logger.info(f"Saved BM25 corpus version {self.settings.bm25_corpus_version} with {len(texts)} documents")
        except Exception as e:
            logger.error(f"Error saving corpus: {e}")
            raise
    
    @traceable(name="load_corpus")
    async def load_corpus(self):
        """Load versioned corpus from Redis."""
        try:
            corpus_data = await self.cache.get_corpus(self.settings.bm25_corpus_version)
            
            if not corpus_data:
                logger.warning(f"Corpus version {self.settings.bm25_corpus_version} not found in cache")
                return False
            
            # Deserialize
            self.tokenized_corpus = pickle.loads(corpus_data)
            
            # Initialize BM25
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            logger.info(f"Loaded BM25 corpus version {self.settings.bm25_corpus_version} with {len(self.tokenized_corpus)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            return False