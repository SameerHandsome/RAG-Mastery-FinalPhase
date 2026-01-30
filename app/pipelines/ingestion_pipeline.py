from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)
from langchain_core.documents import Document
from langsmith import traceable
from app.services.embeddings_service import EmbeddingsService
from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.services.chunker_service import ChunkerService
from app.config import get_settings
from loguru import logger
import asyncio
from pathlib import Path


class IngestionPipeline:
    """Document ingestion pipeline with parallel processing."""
    
    def __init__(
        self,
        embeddings_service: EmbeddingsService,
        vector_store: VectorStore,
        graph_store: GraphStore,
        chunker_service: ChunkerService
    ):
        self.embeddings = embeddings_service
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.chunker = chunker_service
        self.settings = get_settings()
    
    @traceable(name="ingest_documents")
    async def ingest(self, file_paths: list[str]) -> dict:
        """Ingest documents with parallel processing."""
        try:
            # Load documents
            documents = await self._load_documents(file_paths)
            logger.info(f"Loaded {len(documents)} documents")
            
            # Chunk documents
            chunks = await self.chunker.chunk_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Extract texts and metadata
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Save BM25 corpus
            await self.embeddings.save_corpus(texts)
            
            # Parallel embedding generation
            dense_task = self.embeddings.embed_dense(texts)
            sparse_task = self.embeddings.embed_sparse(texts)
            
            dense_embeddings, sparse_embeddings = await asyncio.gather(
                dense_task,
                sparse_task
            )
            
            # Parallel storage
            vector_task = self.vector_store.upsert_documents(
                texts, dense_embeddings, sparse_embeddings, metadatas
            )
            graph_task = self.graph_store.extract_and_add_documents(chunks)
            
            await asyncio.gather(vector_task, graph_task)
            
            return {
                "status": "success",
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "corpus_version": self.settings.bm25_corpus_version
            }
            
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            raise
    
    @traceable(name="load_documents")
    async def _load_documents(self, file_paths: list[str]) -> list[Document]:
        """Load documents from various file types."""
        documents = []
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                
                if not path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Select loader based on file type
                if path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(path))
                elif path.suffix.lower() == '.txt':
                    loader = TextLoader(str(path))
                elif path.suffix.lower() in ['.docx', '.doc']:
                    loader = UnstructuredWordDocumentLoader(str(path))
                else:
                    logger.warning(f"Unsupported file type: {path.suffix}")
                    continue
                
                # Load documents
                docs = loader.load()
                
                # Add file metadata
                for doc in docs:
                    doc.metadata["source"] = path.name
                    doc.metadata["file_type"] = path.suffix.lower()
                
                documents.extend(docs)
                logger.info(f"Loaded {path.name}: {len(docs)} pages")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        return documents