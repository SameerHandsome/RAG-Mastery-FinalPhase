from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration with type-safe environment variable loading."""
    
    # LLM Configuration
    groq_api_key: str
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.7
    llm_timeout: int = 30
    
    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "graph_rag_collection"
    dense_vector_size: int = 384
    
    # Neo4j Configuration
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_query_timeout: int = 40
    
    # Redis Configuration (Upstash REST API)
    redis_url: str = ""  # Legacy support, not required
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    
    # LangSmith Configuration
    langchain_tracing_v2: str = "true"
    langchain_api_key: str
    langchain_project: str = "graph-rag-system"
    
    # BM25 Configuration
    bm25_corpus_version: str = "v1"
    
    # API Configuration
    api_key: str
    
    # Chunking Configuration
    chunk_max_tokens: int = 1000
    chunk_min_tokens: int = 200
    chunk_overlap: int = 50
    
    # Retrieval Configuration
    retrieval_top_k: int = 10
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3
    
    # Cache Configuration
    cache_ttl_hours: int = 1
    embedding_cache_ttl_hours: int = 24
    corpus_cache_ttl_hours: int = 24
    
    # Self-RAG Configuration
    self_rag_max_iterations: int = 3
    self_rag_confidence_threshold: float = 0.7
    
    # Context Compression
    max_context_tokens: int = 8000
    
    # Rate Limiting
    rate_limit_per_minute: int = 100
    rate_limit_tokens_per_minute: int = 50000
    ingest_rate_limit: int = 10
    stream_rate_limit: int = 50
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()