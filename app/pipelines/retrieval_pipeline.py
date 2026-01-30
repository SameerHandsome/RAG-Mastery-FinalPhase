from langsmith import traceable
from app.services.embeddings_service import EmbeddingsService
from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.services.llm_service import LLMService
from app.config import get_settings
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio


class RetrievalPipeline:
    """Retrieval pipeline with query transformation and reranking."""
    
    def __init__(
        self,
        embeddings_service: EmbeddingsService,
        vector_store: VectorStore,
        graph_store: GraphStore,
        llm_service: LLMService
    ):
        self.embeddings = embeddings_service
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.llm = llm_service
        self.settings = get_settings()
    
    @traceable(name="retrieve_documents")
    async def retrieve(self, query: str) -> list[dict]:
        """Retrieve documents with query transformation and reranking."""
        try:
            # Load BM25 corpus
            corpus_loaded = await self.embeddings.load_corpus()
            if not corpus_loaded:
                logger.warning("BM25 corpus not loaded, using dense-only search")
            
            # Generate query variants
            variants = await self.llm.transform_query(query)
            logger.info(f"Generated {len(variants)} query variants")
            
            # Retrieve for each variant
            all_results = []
            
            for variant in variants:
                # Embed query
                dense_emb = await self.embeddings.embed_dense([variant], use_cache=False)
                sparse_emb = await self.embeddings.embed_sparse([variant]) if corpus_loaded else [{}]
                
                # Parallel retrieval
                vector_task = self.vector_store.hybrid_search(
                    dense_emb[0], 
                    sparse_emb[0],
                    top_k=20
                )
                graph_task = self.graph_store.query_graph(variant, top_k=5)
                
                vector_results, graph_results = await asyncio.gather(
                    vector_task, 
                    graph_task,
                    return_exceptions=True
                )
                
                # Handle exceptions
                if isinstance(vector_results, Exception):
                    logger.error(f"Vector search error: {vector_results}")
                    vector_results = []
                
                if isinstance(graph_results, Exception):
                    logger.error(f"Graph search error: {graph_results}")
                    graph_results = []
                
                # Tag vector results with source type
                for result in vector_results:
                    result["metadata"]["retrieval_type"] = "vector_hybrid"
                
                # Merge results from this variant
                variant_results = vector_results + graph_results
                all_results.extend(variant_results)
            
            # Deduplicate by text
            seen_texts = set()
            unique_results = []
            for result in all_results:
                if result["text"] not in seen_texts:
                    seen_texts.add(result["text"])
                    unique_results.append(result)
            
            # Rerank all results
            reranked = await self._rerank(query, unique_results)
            
            logger.info(f"Retrieved and reranked {len(reranked)} unique results")
            return reranked[:self.settings.retrieval_top_k]
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    @traceable(name="rerank_results")
    async def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank results using cosine similarity."""
        if not results:
            return []
        
        try:
            # Embed query and all result texts
            texts = [query] + [r["text"] for r in results]
            embeddings = await self.embeddings.embed_dense(texts, use_cache=False)
            
            # Calculate similarities
            query_emb = np.array(embeddings[0]).reshape(1, -1)
            result_embs = np.array(embeddings[1:])
            
            similarities = cosine_similarity(query_emb, result_embs)[0]
            
            # Update scores
            for i, result in enumerate(results):
                result["score"] = float(similarities[i])
            
            # Sort by similarity
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return results