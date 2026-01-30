from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langsmith import traceable
from openai import RateLimitError
from app.config import get_settings
from loguru import logger


class LLMService:
    """LLM service with retry logic and LangSmith tracing."""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.groq_model,
            openai_api_key=self.settings.groq_api_key,
            openai_api_base=self.settings.groq_base_url,
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout,
            streaming=False
        )
        self.streaming_llm = ChatOpenAI(
            model=self.settings.groq_model,
            openai_api_key=self.settings.groq_api_key,
            openai_api_base=self.settings.groq_base_url,
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout,
            streaming=True
        )
    
    @traceable(name="llm_generate")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(RateLimitError)
    )
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response with retry on rate limits."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    @traceable(name="llm_generate_stream")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(RateLimitError)
    )
    async def generate_stream(self, prompt: str, system_prompt: str = None):
        """Generate streaming response with retry on rate limits."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            async for chunk in self.streaming_llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise
    
    @traceable(name="assess_confidence")
    async def assess_confidence(self, query: str, context: str, answer: str) -> float:
        """Assess confidence in answer given context (0.0-1.0)."""
        prompt = f"""Assess confidence in this answer on a scale of 0.0 to 1.0.

Query: {query}

Context: {context}

Answer: {answer}

Return only a single number between 0.0 and 1.0 representing confidence."""
        
        response = await self.generate(prompt)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    @traceable(name="refine_query")
    async def refine_query(self, original_query: str, previous_context: str) -> str:
        """Refine query based on previous context."""
        prompt = f"""The previous retrieval didn't yield sufficient context. 
Refine this query to be more specific or approach from a different angle.

Original Query: {original_query}

Previous Context: {previous_context}

Return only the refined query, nothing else."""
        
        return await self.generate(prompt)
    
    @traceable(name="transform_query")
    async def transform_query(self, query: str) -> list[str]:
        """Generate 3 query variants for better retrieval."""
        prompt = f"""Generate 3 different variants of this query for better information retrieval.
Each variant should approach the question from a different angle.

Original Query: {query}

Return only the 3 variants, one per line, nothing else."""
        
        response = await self.generate(prompt)
        variants = [v.strip() for v in response.strip().split('\n') if v.strip()]
        return variants[:3] if len(variants) >= 3 else [query, query, query]