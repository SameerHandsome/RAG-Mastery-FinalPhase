import pytest
import asyncio
from langchain_core.documents import Document
from faker import Faker

from app.config import get_settings
from app.services.cache_service import CacheService
from app.services.llm_service import LLMService
from app.services.embeddings_service import EmbeddingsService
from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.services.chunker_service import ChunkerService
from app.pipelines.retrieval_pipeline import RetrievalPipeline
from app.pipelines.generation_pipeline import GenerationPipeline
from app.pipelines.ingestion_pipeline import IngestionPipeline


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings():
    """Get settings."""
    return get_settings()


@pytest.fixture
async def cache_service():
    """Create cache service."""
    service = CacheService()
    await service.clear_all()  # Clear before each test
    return service


@pytest.fixture
def llm_service():
    """Create LLM service."""
    return LLMService()


@pytest.fixture
async def embeddings_service(cache_service):
    """Create embeddings service."""
    return EmbeddingsService(cache_service)


@pytest.fixture
async def vector_store():
    """Create vector store."""
    store = VectorStore()
    await store.setup_collection()
    return store


@pytest.fixture
def graph_store(llm_service):
    """Create graph store."""
    store = GraphStore(llm_service.llm)
    return store


@pytest.fixture
def chunker_service():
    """Create chunker service."""
    return ChunkerService()


@pytest.fixture
async def retrieval_pipeline(embeddings_service, vector_store, graph_store, llm_service):
    """Create retrieval pipeline."""
    return RetrievalPipeline(
        embeddings_service,
        vector_store,
        graph_store,
        llm_service
    )


@pytest.fixture
async def generation_pipeline(llm_service, retrieval_pipeline):
    """Create generation pipeline."""
    return GenerationPipeline(llm_service, retrieval_pipeline)


@pytest.fixture
async def ingestion_pipeline(embeddings_service, vector_store, graph_store, chunker_service):
    """Create ingestion pipeline."""
    return IngestionPipeline(
        embeddings_service,
        vector_store,
        graph_store,
        chunker_service
    )


@pytest.fixture
def sample_documents():
    """Generate sample documents."""
    fake = Faker()
    docs = []
    
    for i in range(5):
        doc = Document(
            page_content=fake.text(max_nb_chars=500),
            metadata={
                "source": f"test_doc_{i}.txt",
                "page": i
            }
        )
        docs.append(doc)
    
    return docs


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning concepts"
    ]