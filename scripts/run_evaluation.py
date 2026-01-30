"""Script to run evaluation on golden dataset."""
import asyncio
from loguru import logger
import sys
import json
from pathlib import Path

sys.path.insert(0, '.')

from app.evaluation import RAGEvaluator
from app.services.llm_service import LLMService
from app.services.embeddings_service import EmbeddingsService
from app.services.cache_service import CacheService
from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.pipelines.retrieval_pipeline import RetrievalPipeline
from app.pipelines.generation_pipeline import GenerationPipeline


async def run_evaluation(dataset_path: str = "data/golden_dataset.json"):
    """Run evaluation on golden dataset."""
    logger.info("Starting evaluation...")
    
    try:
        # Load golden dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        questions = dataset['questions']
        ground_truths = dataset['ground_truths']
        
        logger.info(f"Loaded {len(questions)} test cases")
        
        # Initialize services
        cache_service = CacheService()
        llm_service = LLMService()
        embeddings_service = EmbeddingsService(cache_service)
        vector_store = VectorStore()
        graph_store = GraphStore(llm_service.llm)
        
        retrieval_pipeline = RetrievalPipeline(
            embeddings_service,
            vector_store,
            graph_store,
            llm_service
        )
        
        generation_pipeline = GenerationPipeline(
            llm_service,
            retrieval_pipeline
        )
        
        # Generate answers
        answers = []
        contexts = []
        
        for question in questions:
            logger.info(f"Processing: {question}")
            
            result = await generation_pipeline.generate(question)
            answers.append(result['answer'])
            
            # Extract context texts
            context_texts = [source['text'] for source in result['sources']]
            contexts.append(context_texts)
        
        # Evaluate with RAGAS
        evaluator = RAGEvaluator()
        ragas_results = await evaluator.evaluate_rag(
            questions,
            answers,
            contexts,
            ground_truths
        )
        
        logger.info("="*60)
        logger.info("RAGAS Evaluation Results:")
        logger.info(f"  Faithfulness: {ragas_results.get('faithfulness', 'N/A')}")
        logger.info(f"  Answer Relevancy: {ragas_results.get('answer_relevancy', 'N/A')}")
        logger.info(f"  Context Precision: {ragas_results.get('context_precision', 'N/A')}")
        logger.info(f"  Context Recall: {ragas_results.get('context_recall', 'N/A')}")
        logger.info("="*60)
        
        # Save results
        output_path = Path("evaluation_results.json")
        with open(output_path, 'w') as f:
            json.dump({
                "ragas_results": ragas_results,
                "test_cases": len(questions)
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_evaluation())