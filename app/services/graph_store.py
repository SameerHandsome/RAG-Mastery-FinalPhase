from neo4j import GraphDatabase
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.documents import Document
from langsmith import traceable
from app.config import get_settings
from langchain_core.prompts import PromptTemplate
from loguru import logger
import asyncio

class GraphStore:
    """Neo4j graph store with entity/relationship extraction."""
    
    def __init__(self, llm):
        self.settings = get_settings()
        self.llm = llm
        self.graph = None
        self.transformer = None
        self.qa_chain = None
        
        try:
            # Neo4j connection
            self.graph = Neo4jGraph(
                url=self.settings.neo4j_uri,
                username=self.settings.neo4j_user,
                password=self.settings.neo4j_password
            )
            
            # Graph transformer
            self.transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=[
                    "Person", "Organization", "Location", 
                    "Event", "Concept", "Product"
                ],
                allowed_relationships=[
                    "WORKS_AT", "LOCATED_IN", "PART_OF",
                    "RELATED_TO", "CREATED_BY"
                ]
            )
                    
            # --- REFINED PROMPT TO FIX SYNTAX & EMPTY RESULTS ---
            cypher_prompt = PromptTemplate(
                input_variables=["schema", "question"],
                template="""You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.

RULES FOR SUCCESS:
1.  **Simple Matching**: Do NOT create complex chains like `(a)-[]-(b)-[]-(c)`. 
    -   Instead, match the main entity and find its direct neighbors: `MATCH (n)-[r]-(m) WHERE ...`
2.  **Case Insensitivity**: ALWAYS use `toLower()` for text properties.
    -   `WHERE toLower(n.name) CONTAINS toLower('search_term')`
3.  **Fixing UNWIND**: 
    -   NEVER put a `WHERE` clause immediately after `UNWIND`. 
    -   WRONG: `UNWIND nodes(p) AS n WHERE ...`
    -   RIGHT: `UNWIND nodes(p) AS n WITH n WHERE ...`
4.  **No Backticks**: Do not use backticks on property names (e.g., use `n.name`, not `n.`name``).

Query Strategy:
-   Find nodes whose names vaguely match the user keywords.
-   Return the node names and their descriptions.
-   Limit to 10 results.

Schema:
{schema}

Question: {question}

Return ONLY the Cypher query.

Cypher Query:"""
            )
            
            self.qa_chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=self.graph,
                verbose=True,
                cypher_prompt=cypher_prompt,
                validate_cypher=True,
                return_intermediate_steps=True,
                allow_dangerous_requests=True
            )
            
            logger.info("Neo4j graph store initialized successfully")
            
            # Refresh schema
            try:
                self.graph.refresh_schema()
                logger.info(f"Graph schema loaded: {len(self.graph.structured_schema.get('node_props', {}))} node types")
            except Exception as schema_error:
                logger.warning(f"Could not load graph schema: {schema_error}")
            
        except Exception as e:
            logger.warning(f"Neo4j initialization failed: {e}")
            logger.warning("Continuing without graph database (vector search only)")
            self.graph = None
    
    @traceable(name="graph_extract_entities")
    async def extract_and_add_documents(self, documents: list[Document], batch_size: int = 5):
        """Extract entities/relationships and add to graph."""
        if not self.graph:
            return
            
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    graph_docs = self.transformer.convert_to_graph_documents(batch)
                    self.graph.add_graph_documents(graph_docs)
                    logger.info(f"Processed batch {i // batch_size + 1} ({len(batch)} documents)")
                except Exception as batch_error:
                    logger.warning(f"Graph extraction failed for batch {i // batch_size + 1}: {batch_error}")
                    continue
            logger.info(f"Completed graph processing for {len(documents)} documents")
        except Exception as e:
            logger.error(f"Graph extraction error: {e}")
    
    @traceable(name="graph_query")
    async def query_graph(self, query: str, top_k: int = 5) -> list[dict]:
        """Query graph with fallback to simple keyword search."""
        if not self.graph or not self.qa_chain:
            return []
            
        try:
            # 1. Try LLM Generation
            result = await asyncio.wait_for(
                asyncio.to_thread(self._sync_query, query),
                timeout=self.settings.neo4j_query_timeout
            )
            
            if result:
                return [{
                    "text": result,
                    "metadata": {"source": "graph_qa"},
                    "score": 1.0 
                }]
            
            # 2. Fallback
            logger.debug("QA chain empty, switching to keyword fallback")
            return await self._direct_keyword_search(query, top_k)
            
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"Graph query failed: {e}")
            return await self._direct_keyword_search(query, top_k)
    
    async def _direct_keyword_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Robust Fallback Search."""
        try:
            keywords = [w.lower() for w in query.split() if len(w) > 3][:5]
            if not keywords: return []
            
            # Simple, fail-safe query
            cypher = """
                MATCH (n)
                WHERE any(keyword IN $keywords WHERE toLower(n.name) CONTAINS keyword)
                RETURN n.name as entity, coalesce(n.description, '') as description
                LIMIT $limit
            """
            
            result = await asyncio.to_thread(
                self.graph.query, cypher, {"keywords": keywords, "limit": top_k}
            )
            
            formatted = []
            for row in result:
                text = f"Graph Entity: {row['entity']}. {row['description']}"
                formatted.append({
                    "text": text,
                    "metadata": {"source": "graph_fallback"},
                    "score": 0.5
                })
            return formatted
            
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []
    
    def _sync_query(self, query: str) -> str:
        """Synchronous wrapper."""
        try:
            response = self.qa_chain.invoke({"query": query})
            
            # Debug Log
            if isinstance(response, dict) and "intermediate_steps" in response:
                steps = response["intermediate_steps"]
                if steps:
                    logger.debug(f"Generated Cypher: {steps[0].get('query', 'N/A')}")

            result = response.get("result", "") if isinstance(response, dict) else str(response)
            
            if not result or "i don't know" in result.lower():
                return ""
            return result
        except Exception as e:
            logger.warning(f"GraphQA Chain Syntax/Runtime Error: {e}")
            return ""

    async def clear_graph(self):
        if self.graph:
            self.graph.query("MATCH (n) DETACH DELETE n")