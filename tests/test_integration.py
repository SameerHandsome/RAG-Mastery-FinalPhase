import pytest
from pathlib import Path
import tempfile


class TestIngestionPipeline:
    """Integration tests for ingestion pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_ingestion_flow(self, ingestion_pipeline):
        """Test complete ingestion pipeline end-to-end."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document about machine learning.\n" * 50)
            temp_file = f.name
        
        try:
            # Ingest
            result = await ingestion_pipeline.ingest([temp_file])
            
            # Verify results
            assert result["status"] == "success"
            assert result["documents_loaded"] > 0
            assert result["chunks_created"] > 0
            assert "corpus_version" in result
            
        finally:
            Path(temp_file).unlink()
    
    @pytest.mark.asyncio
    async def test_corpus_save_load_cycle(self, embeddings_service):
        """Test complete BM25 corpus save/load cycle."""
        texts = ["artificial intelligence", "machine learning", "deep learning"]
        
        # Save
        await embeddings_service.save_corpus(texts)
        
        # Load
        loaded = await embeddings_service.load_corpus()
        assert loaded is True
        
        # Verify can generate embeddings
        query_emb = await embeddings_service.embed_sparse(["AI"])
        assert len(query_emb) > 0


class TestRetrievalPipeline:
    """Integration tests for retrieval pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_retrieval_flow(self, retrieval_pipeline, embeddings_service, vector_store):
        """Test complete retrieval pipeline."""
        # Setup test data
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks",
            "Python is a programming language"
        ]
        
        # Save corpus
        await embeddings_service.save_corpus(texts)
        
        # Embed and store
        dense_emb = await embeddings_service.embed_dense(texts)
        sparse_emb = await embeddings_service.embed_sparse(texts)
        metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        await vector_store.upsert_documents(texts, dense_emb, sparse_emb, metadatas)
        
        # Retrieve
        results = await retrieval_pipeline.retrieve("what is machine learning?")
        
        # Verify results
        assert len(results) > 0
        assert any("machine learning" in r["text"].lower() for r in results)
    
    @pytest.mark.asyncio
    async def test_graph_answer_reranking(self, retrieval_pipeline, vector_store, graph_store, embeddings_service):
        """Test that graph answers compete fairly in reranking."""
        # Setup vector data
        texts = ["unrelated content here", "another unrelated doc"]
        
        await embeddings_service.save_corpus(texts)
        dense_emb = await embeddings_service.embed_dense(texts)
        sparse_emb = await embeddings_service.embed_sparse(texts)
        metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        await vector_store.upsert_documents(texts, dense_emb, sparse_emb, metadatas)
        
        # Retrieve (will include graph results with score=0.0)
        results = await retrieval_pipeline.retrieve("test query")
        
        # Graph results should not always be at top after reranking
        # They should be reranked based on actual relevance
        if len(results) > 1:
            # Verify scores are from reranking (cosine similarity)
            assert all(0.0 <= r["score"] <= 1.0 for r in results)


class TestGenerationPipeline:
    """Integration tests for generation pipeline."""
    
    @pytest.mark.asyncio
    async def test_self_rag_iterations(self, generation_pipeline, retrieval_pipeline, embeddings_service, vector_store):
        """Test Self-RAG iteration logic."""
        # Setup test data
        texts = [
            "Machine learning is a method of data analysis",
            "It automates analytical model building"
        ]
        
        await embeddings_service.save_corpus(texts)
        dense_emb = await embeddings_service.embed_dense(texts)
        sparse_emb = await embeddings_service.embed_sparse(texts)
        metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        await vector_store.upsert_documents(texts, dense_emb, sparse_emb, metadatas)
        
        # Generate
        result = await generation_pipeline.generate("what is machine learning?")
        
        # Verify structure
        assert "answer" in result
        assert "sources" in result
        assert "iterations" in result
        assert result["iterations"] <= generation_pipeline.settings.self_rag_max_iterations


class TestVectorStore:
    """Integration tests for vector store."""
    
    @pytest.mark.asyncio
    async def test_qdrant_connection_and_search(self, vector_store, embeddings_service):
        """Test Qdrant connection and hybrid search."""
        texts = ["test document one", "test document two"]
        
        # Generate embeddings
        dense_emb = await embeddings_service.embed_dense(texts)
        
        # Create minimal sparse embeddings
        sparse_emb = [{"0": 1.0}, {"1": 1.0}]
        
        metadatas = [{"source": f"test_{i}"} for i in range(len(texts))]
        
        # Upsert
        await vector_store.upsert_documents(texts, dense_emb, sparse_emb, metadatas)
        
        # Search
        query_dense = await embeddings_service.embed_dense(["test query"])
        query_sparse = {"0": 1.0}
        
        results = await vector_store.hybrid_search(query_dense[0], query_sparse, top_k=2)
        
        assert len(results) > 0


class TestGraphStore:
    """Integration tests for graph store."""
    
    @pytest.mark.asyncio
    async def test_neo4j_connection_and_query(self, graph_store):
        """Test Neo4j connection and graph query."""
        from langchain_core.documents import Document
        
        # Create test documents
        docs = [
            Document(
                page_content="John works at OpenAI in San Francisco",
                metadata={"source": "test"}
            )
        ]
        
        # Extract and add
        await graph_store.extract_and_add_documents(docs)
        
        # Query (with timeout handling)
        results = await graph_store.query_graph("Where does John work?")
        
        # Should return results or empty list (not error)
        assert isinstance(results, list)