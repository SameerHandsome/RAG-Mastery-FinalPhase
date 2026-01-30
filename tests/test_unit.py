import pytest
from unittest.mock import Mock, patch
from openai import RateLimitError


class TestLLMService:
    """Unit tests for LLM service."""
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, llm_service):
        """Test retry logic on rate limit errors."""
        with patch.object(llm_service.llm, 'ainvoke') as mock_invoke:
            # First two calls fail, third succeeds
            mock_invoke.side_effect = [
                RateLimitError("Rate limit"),
                RateLimitError("Rate limit"),
                Mock(content="Success")
            ]
            
            result = await llm_service.generate("test prompt")
            assert result == "Success"
            assert mock_invoke.call_count == 3
    
    @pytest.mark.asyncio
    async def test_query_transformation(self, llm_service):
        """Test query transformation generates variants."""
        variants = await llm_service.transform_query("What is AI?")
        assert len(variants) == 3
        assert all(isinstance(v, str) for v in variants)


class TestCacheService:
    """Unit tests for cache service."""
    
    @pytest.mark.asyncio
    async def test_query_cache_hit(self, cache_service):
        """Test cache hit scenario."""
        query = "test query"
        response = "test response"
        
        await cache_service.set_query_response(query, response)
        cached = await cache_service.get_query_response(query)
        
        assert cached == response
    
    @pytest.mark.asyncio
    async def test_query_cache_miss(self, cache_service):
        """Test cache miss scenario."""
        cached = await cache_service.get_query_response("nonexistent query")
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_embedding_cache(self, cache_service):
        """Test embedding caching."""
        embedding = [0.1, 0.2, 0.3]
        text_hash = "test_hash"
        
        await cache_service.set_embedding(text_hash, embedding)
        cached = await cache_service.get_embedding(text_hash)
        
        assert cached == embedding


class TestEmbeddingsService:
    """Unit tests for embeddings service."""
    
    @pytest.mark.asyncio
    async def test_dense_embedding_generation(self, embeddings_service):
        """Test dense embedding generation."""
        texts = ["hello world", "test text"]
        embeddings = await embeddings_service.embed_dense(texts, use_cache=False)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384  # all-MiniLM-L6-v2 dimension
    
    @pytest.mark.asyncio
    async def test_corpus_save_and_load(self, embeddings_service, cache_service):
        """Test BM25 corpus versioning."""
        texts = ["document one", "document two", "document three"]
        
        # Save corpus
        await embeddings_service.save_corpus(texts)
        
        # Create new service instance
        new_service = EmbeddingsService(cache_service)
        
        # Load corpus
        loaded = await new_service.load_corpus()
        assert loaded is True
        
        # Verify BM25 initialized
        assert new_service.bm25 is not None
        assert len(new_service.tokenized_corpus) == 3
    
    @pytest.mark.asyncio
    async def test_sparse_embedding_consistency(self, embeddings_service):
        """Test sparse score consistency across runs."""
        texts = ["machine learning", "deep learning", "neural networks"]
        
        # Save corpus
        await embeddings_service.save_corpus(texts)
        
        # Generate sparse embeddings twice
        query = ["machine learning"]
        sparse1 = await embeddings_service.embed_sparse(query)
        sparse2 = await embeddings_service.embed_sparse(query)
        
        # Should be identical
        assert sparse1 == sparse2


class TestChunkerService:
    """Unit tests for chunker service."""
    
    @pytest.mark.asyncio
    async def test_chunking_validation(self, chunker_service, sample_documents):
        """Test chunking creates valid chunks."""
        chunks = await chunker_service.chunk_documents(sample_documents)
        
        assert len(chunks) > 0
        
        # Verify all chunks meet minimum size
        for chunk in chunks:
            word_count = len(chunk.page_content.split())
            assert word_count >= 50
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, chunker_service, sample_documents):
        """Test metadata is preserved in chunks."""
        chunks = await chunker_service.chunk_documents(sample_documents)
        
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata


class TestRetrievalPipeline:
    """Unit tests for retrieval pipeline."""
    
    @pytest.mark.asyncio
    async def test_reranking_logic(self, retrieval_pipeline):
        """Test manual reranking with cosine similarity."""
        query = "machine learning"
        
        # Mock results
        results = [
            {"text": "machine learning algorithms", "score": 0.5, "metadata": {}},
            {"text": "unrelated content here", "score": 0.8, "metadata": {}},
            {"text": "deep learning neural networks", "score": 0.6, "metadata": {}}
        ]
        
        # Rerank
        reranked = await retrieval_pipeline._rerank(query, results)
        
        # Most relevant should be first
        assert "machine learning" in reranked[0]["text"].lower()


class TestGenerationPipeline:
    """Unit tests for generation pipeline."""
    
    @pytest.mark.asyncio
    async def test_context_compression(self, generation_pipeline):
        """Test context compression stays within token limit."""
        # Create large context
        large_results = [
            {"text": "test " * 1000, "score": 0.9, "metadata": {}}
            for _ in range(20)
        ]
        
        context = generation_pipeline._build_context(large_results)
        compressed = await generation_pipeline._compress_context(context, large_results)
        
        # Verify within limit
        tokens = generation_pipeline.tokenizer.encode(compressed)
        assert len(tokens) <= generation_pipeline.settings.max_context_tokens