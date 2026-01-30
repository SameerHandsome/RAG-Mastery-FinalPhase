from semantic_text_splitter import TextSplitter
from langchain_core.documents import Document
from langsmith import traceable
from app.config import get_settings
from loguru import logger


class ChunkerService:
    """Semantic text chunking service using Rust-based splitter."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Semantic splitter (Rust-based, 10x faster)
        self.splitter = TextSplitter(
            capacity=self.settings.chunk_max_tokens,
            overlap=self.settings.chunk_overlap
        )
    
    @traceable(name="chunk_documents")
    async def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk documents semantically with metadata preservation."""
        try:
            chunked_docs = []
            
            for doc in documents:
                # Split text
                chunks = self.splitter.chunks(doc.page_content)
                
                # Filter chunks (minimum ~200 tokens = ~50 words)
                valid_chunks = [
                    chunk for chunk in chunks 
                    if len(chunk.split()) >= 50
                ]
                
                # Create documents with metadata
                for i, chunk in enumerate(valid_chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "total_chunks": len(valid_chunks)
                        }
                    )
                    chunked_docs.append(chunk_doc)
            
            logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Chunking error: {e}")
            raise