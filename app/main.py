from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from langsmith import traceable
from loguru import logger
from pathlib import Path
import tempfile
import sys

from app.config import get_settings
from app.services.llm_service import LLMService
from app.services.embeddings_service import EmbeddingsService
from app.services.cache_service import CacheService
from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.services.chunker_service import ChunkerService
from app.pipelines.retrieval_pipeline import RetrievalPipeline
from app.pipelines.generation_pipeline import GenerationPipeline
from app.pipelines.ingestion_pipeline import IngestionPipeline

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Initialize app
app = FastAPI(title="Graph RAG System", version="1.0.0")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Settings
settings = get_settings()

# Initialize services (lazy loading)
_services = {}


def get_services():
    """Get or initialize services."""
    if not _services:
        # Core services
        cache_service = CacheService()
        llm_service = LLMService()
        embeddings_service = EmbeddingsService(cache_service)
        vector_store = VectorStore()
        graph_store = GraphStore(llm_service.llm)
        chunker_service = ChunkerService()
        
        # Pipelines
        retrieval_pipeline = RetrievalPipeline(
            embeddings_service, vector_store, graph_store, llm_service
        )
        generation_pipeline = GenerationPipeline(llm_service, retrieval_pipeline)
        ingestion_pipeline = IngestionPipeline(
            embeddings_service, vector_store, graph_store, chunker_service
        )
        
        _services.update({
            "cache": cache_service,
            "llm": llm_service,
            "embeddings": embeddings_service,
            "vector_store": vector_store,
            "graph_store": graph_store,
            "chunker": chunker_service,
            "retrieval": retrieval_pipeline,
            "generation": generation_pipeline,
            "ingestion": ingestion_pipeline
        })
    
    return _services


# API key dependency
async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header."""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float = None
    iterations: int = None
    cached: bool = False


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    corpus_version: str


# Routes
@app.get("/health")
@traceable(name="health_check")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/corpus/version")
@traceable(name="get_corpus_version")
async def get_corpus_version(api_key: str = Depends(verify_api_key)):
    """Get current BM25 corpus version."""
    return {"version": settings.bm25_corpus_version}


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("10/minute")
@traceable(name="ingest_endpoint")
async def ingest_documents(
    request: Request,  # Required for rate limiter
    files: list[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Ingest documents endpoint."""
    try:
        services = get_services()
        
        file_paths = []
        temp_dir = tempfile.mkdtemp()
        
        for file in files:
            file_path = Path(temp_dir) / file.filename
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            file_paths.append(str(file_path))
        
        # Ingest
        result = await services["ingestion"].ingest(file_paths)
        
        # Cleanup
        for file_path in file_paths:
            Path(file_path).unlink()
        Path(temp_dir).rmdir()
        
        return IngestResponse(**result)
        
    except Exception as e:
        logger.error(f"Ingestion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
@limiter.limit("100/minute")
@traceable(name="query_endpoint")
async def query(
    request: Request,  # Required for rate limiter
    request_body: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Query endpoint with caching."""
    try:
        services = get_services()
        
        # Check cache
        if request_body.use_cache:
            cached_response = await services["cache"].get_query_response(request_body.query)
            if cached_response:
                import json
                response_data = json.loads(cached_response)
                response_data["cached"] = True
                return QueryResponse(**response_data)
        
        # Generate answer
        result = await services["generation"].generate(request_body.query)
        
        # Cache response
        if request_body.use_cache:
            import json
            await services["cache"].set_query_response(
                request_body.query, 
                json.dumps(result)
            )
        
        return QueryResponse(**result, cached=False)
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
@limiter.limit("50/minute")
@traceable(name="query_stream_endpoint")
async def query_stream(
    request: Request,  # Required for rate limiter
    request_body: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Streaming query endpoint."""
    try:
        services = get_services()
        
        async def event_generator():
            try:
                async for chunk in services["generation"].generate_stream(request_body.query):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: Error: {str(e)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Graph RAG System...")
    
    # Initialize services
    services = get_services()
    
    # Setup vector store
    await services["vector_store"].setup_collection()
    
    logger.info("Graph RAG System started successfully")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)