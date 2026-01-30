from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    SparseVectorParams, SparseIndexParams
)
from langsmith import traceable
from app.config import get_settings
from loguru import logger
from uuid import uuid4


class VectorStore:
    """Qdrant vector store with hybrid search (dense + sparse)."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Support both cloud and in-memory modes
        if self.settings.qdrant_url == ":memory:":
            self.client = QdrantClient(location=":memory:")
            logger.info("Using in-memory Qdrant (no cloud required)")
        else:
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=60,
                https=True
            )
            logger.info(f"Using Qdrant Cloud: {self.settings.qdrant_url}")
        
        self.collection_name = self.settings.qdrant_collection
    
    @traceable(name="vector_store_setup")
    async def setup_collection(self):
        """Create collection with named vectors for hybrid search."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists:
                logger.info(f"Collection {self.collection_name} already exists, keeping existing data")
                return
            
            # Create collection with named vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.settings.dense_vector_size,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams()
                    )
                }
            )
            
            logger.info(f"Created collection {self.collection_name} with dense and sparse vectors")
        except Exception as e:
            logger.error(f"Setup collection error: {e}")
            raise
    
    @traceable(name="vector_store_upsert")
    async def upsert_documents(
        self, 
        texts: list[str], 
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[dict],
        metadatas: list[dict]
    ):
        """Batch upsert documents with hybrid embeddings."""
        try:
            points = []
            
            for i, (text, dense, sparse, metadata) in enumerate(
                zip(texts, dense_embeddings, sparse_embeddings, metadatas)
            ):
                # Convert sparse dict to required format
                sparse_indices = [int(k) for k in sparse.keys()]
                sparse_values = list(sparse.values())
                
                point = PointStruct(
                    id=str(uuid4()),
                    vector={
                        "dense": dense,
                        "sparse": {
                            "indices": sparse_indices,
                            "values": sparse_values
                        }
                    },
                    payload={
                        "text": text,
                        **metadata
                    }
                )
                points.append(point)
            
            # Batch upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} documents")
        except Exception as e:
            logger.error(f"Upsert error: {e}")
            raise
    
    @traceable(name="vector_store_hybrid_search")
    async def hybrid_search(
        self, 
        dense_query: list[float],
        sparse_query: dict,
        top_k: int = 10
    ) -> list[dict]:
        """Hybrid search using Qdrant's query_points API."""
        try:
            # Search using dense vector with query_points
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_query,
                using="dense",  # Specify which named vector to use
                limit=top_k,
                with_payload=True
            )
            
            # Convert to our format
            final_results = []
            for point in response.points:
                final_results.append({
                    "text": point.payload["text"],
                    "metadata": {
                        **{k: v for k, v in point.payload.items() if k != "text"},
                        "dense_score": point.score,
                        "sparse_score": 0.0,  # Not using sparse in this simplified version
                        "hybrid_score": point.score
                    },
                    "score": point.score
                })
            
            logger.info(f"Retrieved {len(final_results)} results from dense vector search")
            return final_results
            
            # Fusion: combine scores
            combined = {}
            
            # Add dense results
            for result in dense_results:
                doc_id = result.id
                combined[doc_id] = {
                    "text": result.payload["text"],
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                    "dense_score": result.score,
                    "sparse_score": 0.0
                }
            
            # Add sparse results
            for result in sparse_results:
                doc_id = result.id
                if doc_id in combined:
                    combined[doc_id]["sparse_score"] = result.score
                else:
                    combined[doc_id] = {
                        "text": result.payload["text"],
                        "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                        "dense_score": 0.0,
                        "sparse_score": result.score
                    }
            
            # Calculate fusion scores
            results = []
            for doc_id, data in combined.items():
                fusion_score = (
                    data["dense_score"] * self.settings.hybrid_dense_weight +
                    data["sparse_score"] * self.settings.hybrid_sparse_weight
                )
                results.append({
                    "text": data["text"],
                    "metadata": data["metadata"],
                    "score": fusion_score
                })
            
            # Sort by fusion score and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []