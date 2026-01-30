from langsmith import traceable
from app.services.llm_service import LLMService
from app.pipelines.retrieval_pipeline import RetrievalPipeline
from app.config import get_settings
from loguru import logger
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class GenerationPipeline:
    """Generation pipeline with Self-RAG and context compression."""
    
    def __init__(
        self,
        llm_service: LLMService,
        retrieval_pipeline: RetrievalPipeline
    ):
        self.llm = llm_service
        self.retrieval = retrieval_pipeline
        self.settings = get_settings()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @traceable(name="generate_answer")
    async def generate(self, query: str) -> dict:
        """Generate answer with Self-RAG loop."""
        try:
            iteration = 0
            current_query = query
            
            while iteration < self.settings.self_rag_max_iterations:
                # Retrieve
                results = await self.retrieval.retrieve(current_query)
                
                if not results:
                    return {
                        "answer": "I couldn't find relevant information to answer your query.",
                        "sources": [],
                        "iterations": iteration + 1
                    }
                
                # Extract context
                context = self._build_context(results)
                
                # Compress context
                compressed_context = await self._compress_context(context, results)
                
                # Generate answer
                answer = await self._generate_answer(query, compressed_context)
                
                # Assess confidence
                confidence = await self.llm.assess_confidence(query, compressed_context, answer)
                
                logger.info(f"Self-RAG iteration {iteration + 1}, confidence: {confidence}")
                
                # Check confidence threshold
                if confidence >= self.settings.self_rag_confidence_threshold:
                    return {
                        "answer": answer,
                        "sources": self._format_sources(results),
                        "confidence": confidence,
                        "iterations": iteration + 1
                    }
                
                # Last iteration - return anyway
                if iteration == self.settings.self_rag_max_iterations - 1:
                    return {
                        "answer": answer,
                        "sources": self._format_sources(results),
                        "confidence": confidence,
                        "iterations": iteration + 1
                    }
                
                # Refine query for next iteration
                current_query = await self.llm.refine_query(query, compressed_context)
                iteration += 1
            
            # Fallback
            return {
                "answer": "Unable to generate a confident answer.",
                "sources": [],
                "iterations": iteration
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    @traceable(name="generate_answer_stream")
    async def generate_stream(self, query: str):
        """Generate streaming answer with Self-RAG."""
        try:
            # Single iteration for streaming
            results = await self.retrieval.retrieve(query)
            
            if not results:
                yield "I couldn't find relevant information to answer your query."
                return
            
            # Build and compress context
            context = self._build_context(results)
            compressed_context = await self._compress_context(context, results)
            
            # Stream answer
            system_prompt = "Answer the question based on the provided context. Be concise and accurate."
            prompt = f"Context:\n{compressed_context}\n\nQuestion: {query}\n\nAnswer:"
            
            async for chunk in self.llm.generate_stream(prompt, system_prompt):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"Error: {str(e)}"
    
    def _build_context(self, results: list[dict]) -> str:
        """Build context from results."""
        context_parts = []
        for i, result in enumerate(results):
            source = result["metadata"].get("source", "unknown")
            context_parts.append(f"[{i+1}] (Source: {source})\n{result['text']}")
        return "\n\n".join(context_parts)
    
    @traceable(name="compress_context")
    async def _compress_context(self, context: str, results: list[dict]) -> str:
        """Compress context to fit token limit."""
        tokens = self.tokenizer.encode(context)
        
        if len(tokens) <= self.settings.max_context_tokens:
            return context
        
        logger.info(f"Compressing context from {len(tokens)} to {self.settings.max_context_tokens} tokens")
        
        # Remove redundant chunks
        filtered_results = await self._remove_redundant(results)
        
        # Build compressed context
        compressed = []
        current_tokens = 0
        
        for i, result in enumerate(filtered_results):
            chunk_text = f"[{i+1}] {result['text']}"
            chunk_tokens = len(self.tokenizer.encode(chunk_text))
            
            if current_tokens + chunk_tokens > self.settings.max_context_tokens:
                break
            
            compressed.append(chunk_text)
            current_tokens += chunk_tokens
        
        return "\n\n".join(compressed)
    
    async def _remove_redundant(self, results: list[dict], threshold: float = 0.9) -> list[dict]:
        """Remove highly similar chunks."""
        if len(results) <= 1:
            return results
        
        try:
            from app.services.embeddings_service import EmbeddingsService
            from app.services.cache_service import CacheService
            
            # Quick embedding for similarity check
            cache = CacheService()
            embeddings_service = EmbeddingsService(cache)
            
            texts = [r["text"] for r in results]
            embeddings = await embeddings_service.embed_dense(texts, use_cache=False)
            
            # Calculate similarity matrix
            emb_matrix = np.array(embeddings)
            similarities = cosine_similarity(emb_matrix)
            
            # Keep diverse chunks
            keep_indices = [0]  # Always keep first (highest scored)
            
            for i in range(1, len(results)):
                # Check if similar to any kept chunk
                similar_to_kept = any(
                    similarities[i][j] > threshold 
                    for j in keep_indices
                )
                
                if not similar_to_kept:
                    keep_indices.append(i)
            
            return [results[i] for i in keep_indices]
            
        except Exception as e:
            logger.error(f"Redundancy removal error: {e}")
            return results
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer from context."""
        system_prompt = "Answer the question based on the provided context. Be concise and accurate."
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        return await self.llm.generate(prompt, system_prompt)
    
    def _format_sources(self, results: list[dict]) -> list[dict]:
        """Format sources for response with retrieval type labels."""
        sources = []
        for result in results:
            metadata = result["metadata"].copy()
            
            # Add retrieval type label
            if "retrieval_type" in metadata:
                retrieval_type = metadata["retrieval_type"]
            elif metadata.get("source") == "graph":
                retrieval_type = "graph_database"
            else:
                retrieval_type = "vector_hybrid"
            
            sources.append({
                "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "metadata": metadata,
                "score": result["score"],
                "retrieval_type": retrieval_type
            })
        return sources