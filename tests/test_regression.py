import pytest
import time
from app.evaluation import RAGEvaluator


class TestRegressionSuite:
    """Regression tests with golden dataset."""
    
    @pytest.fixture
    def golden_dataset(self):
        """Golden dataset for regression testing."""
        return {
            "questions": [
                "What is machine learning?",
                "How do neural networks work?",
                "What is deep learning?"
            ],
            "ground_truths": [
                "Machine learning is a method of data analysis that automates analytical model building.",
                "Neural networks are computing systems inspired by biological neural networks.",
                "Deep learning is a subset of machine learning based on artificial neural networks."
            ]
        }
    
    @pytest.mark.asyncio
    async def test_ragas_score_threshold(self, generation_pipeline, golden_dataset):
        """Test that RAGAS scores meet threshold."""
        # This is a simplified test - in production, you'd have full context
        evaluator = RAGEvaluator()
        
        # Mock answers and contexts for testing
        answers = [
            "Machine learning automates data analysis",
            "Neural networks are inspired by biological systems",
            "Deep learning uses neural networks"
        ]
        
        contexts = [
            ["Machine learning is automated analytical modeling"],
            ["Neural networks mimic biological computation"],
            ["Deep learning is a neural network subset"]
        ]
        
        # Evaluate
        results = await evaluator.evaluate_rag(
            golden_dataset["questions"],
            answers,
            contexts,
            golden_dataset["ground_truths"]
        )
        
        # Check thresholds
        assert results.get("faithfulness", 0) >= 0.85
        assert results.get("answer_relevancy", 0) >= 0.85
    
    @pytest.mark.asyncio
    async def test_latency_sla(self, generation_pipeline, embeddings_service, vector_store):
        """Test that p95 latency is under 5 seconds."""
        # Setup minimal data
        texts = ["test content"] * 10
        
        await embeddings_service.save_corpus(texts)
        dense_emb = await embeddings_service.embed_dense(texts)
        sparse_emb = await embeddings_service.embed_sparse(texts)
        metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        await vector_store.upsert_documents(texts, dense_emb, sparse_emb, metadatas)
        
        # Measure latencies
        latencies = []
        queries = ["test query " + str(i) for i in range(10)]
        
        for query in queries:
            start = time.time()
            await generation_pipeline.generate(query)
            latency = time.time() - start
            latencies.append(latency)
        
        # Calculate p95
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        
        assert p95 < 5.0, f"p95 latency {p95}s exceeds 5s SLA"
    
    @pytest.mark.asyncio
    async def test_bm25_score_stability(self, embeddings_service):
        """Test BM25 scores are stable across runs."""
        texts = ["machine learning", "deep learning", "neural networks"]
        
        # Save corpus
        await embeddings_service.save_corpus(texts)
        
        # Generate sparse embeddings multiple times
        query = ["machine learning algorithms"]
        
        scores_run1 = await embeddings_service.embed_sparse(query)
        scores_run2 = await embeddings_service.embed_sparse(query)
        scores_run3 = await embeddings_service.embed_sparse(query)
        
        # All runs should produce identical scores
        assert scores_run1 == scores_run2 == scores_run3
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self, generation_pipeline, cache_service, embeddings_service, vector_store):
        """Test cache hit rate is above 50%."""
        # Setup data
        texts = ["test content"] * 5
        
        await embeddings_service.save_corpus(texts)
        dense_emb = await embeddings_service.embed_dense(texts)
        sparse_emb = await embeddings_service.embed_sparse(texts)
        metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        await vector_store.upsert_documents(texts, dense_emb, sparse_emb, metadatas)
        
        # Make queries
        queries = ["test query"] * 10  # Same query repeated
        
        cache_hits = 0
        for query in queries:
            # Check cache before generation
            cached = await cache_service.get_query_response(query)
            
            if cached:
                cache_hits += 1
            else:
                # Generate and cache
                result = await generation_pipeline.generate(query)
                import json
                await cache_service.set_query_response(query, json.dumps(result))
        
        hit_rate = cache_hits / len(queries)
        assert hit_rate >= 0.5, f"Cache hit rate {hit_rate} below 50% threshold"


class TestRetrievalMetrics:
    """Test retrieval quality metrics."""
    
    @pytest.mark.asyncio
    async def test_retrieval_precision_recall(self):
        """Test custom retrieval metrics calculation."""
        evaluator = RAGEvaluator()
        
        # Mock retrieved and relevant docs
        retrieved = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"]
        ]
        
        relevant = [
            ["doc1", "doc3"],
            ["doc5", "doc7"]
        ]
        
        # Evaluate
        metrics = evaluator.evaluate_retrieval(retrieved, relevant, k_values=[3])
        
        # Verify metrics exist
        assert "precision@3" in metrics
        assert "recall@3" in metrics
        assert "hit@3" in metrics
        assert "mrr" in metrics
        assert "ndcg@10" in metrics
        
        # Verify reasonable values
        assert 0 <= metrics["precision@3"] <= 1
        assert 0 <= metrics["recall@3"] <= 1