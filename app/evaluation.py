from langsmith import traceable
from loguru import logger
from app.config import get_settings
from langchain_openai import ChatOpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class RAGEvaluator:
    """Evaluation framework with custom metrics (Groq-compatible)."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Use your Groq LLM for evaluations
        self.llm = ChatOpenAI(
            model=self.settings.groq_model,
            openai_api_key=self.settings.groq_api_key,
            openai_api_base=self.settings.groq_base_url,
            temperature=0.0  # Deterministic for evaluation
        )
        
        # Embedding model for similarity calculations
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    @traceable(name="evaluate_rag")
    async def evaluate_rag(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str]
    ) -> dict:
        """Evaluate RAG system using custom metrics with Groq."""
        try:
            # Calculate metrics
            faithfulness_scores = []
            answer_relevancy_scores = []
            context_precision_scores = []
            context_recall_scores = []
            
            for i, (question, answer, context_list, ground_truth) in enumerate(
                zip(questions, answers, contexts, ground_truths)
            ):
                logger.info(f"Evaluating question {i+1}/{len(questions)}")
                
                # Faithfulness: Is answer grounded in context?
                faith_score = await self._evaluate_faithfulness(answer, context_list)
                faithfulness_scores.append(faith_score)
                
                # Answer Relevancy: Does answer address the question?
                relevancy_score = await self._evaluate_answer_relevancy(question, answer)
                answer_relevancy_scores.append(relevancy_score)
                
                # Context Precision: Are retrieved contexts relevant?
                precision_score = self._evaluate_context_precision(question, context_list)
                context_precision_scores.append(precision_score)
                
                # Context Recall: Does context cover ground truth?
                recall_score = self._evaluate_context_recall(context_list, ground_truth)
                context_recall_scores.append(recall_score)
            
            results = {
                "faithfulness": float(np.mean(faithfulness_scores)),
                "answer_relevancy": float(np.mean(answer_relevancy_scores)),
                "context_precision": float(np.mean(context_precision_scores)),
                "context_recall": float(np.mean(context_recall_scores)),
                "individual_scores": {
                    "faithfulness": faithfulness_scores,
                    "answer_relevancy": answer_relevancy_scores,
                    "context_precision": context_precision_scores,
                    "context_recall": context_recall_scores
                }
            }
            
            logger.info(f"RAGAS-style Evaluation Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"RAG evaluation error: {e}")
            raise
    
    async def _evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Check if answer is grounded in provided context."""
        context_text = "\n".join(contexts[:3])  # Use top 3 contexts
        
        prompt = f"""Given the context and answer below, evaluate if the answer is faithful to the context.
        
Context:
{context_text}

Answer:
{answer}

Is this answer fully supported by the context? Rate from 0.0 (not supported) to 1.0 (fully supported).
Consider:
- Does the answer contain claims not in the context?
- Is the answer factually consistent with the context?

Return ONLY a number between 0.0 and 1.0.
"""
        
        try:
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            # Fallback: Similarity-based
            return self._similarity_score(answer, context_text)
    
    async def _evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """Check if answer is relevant to the question."""
        prompt = f"""Given the question and answer below, evaluate the relevance of the answer.

Question:
{question}

Answer:
{answer}

Does the answer directly address the question? Rate from 0.0 (irrelevant) to 1.0 (perfectly relevant).

Return ONLY a number between 0.0 and 1.0.
"""
        
        try:
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            # Fallback: Similarity-based
            return self._similarity_score(question, answer)
    
    def _evaluate_context_precision(self, question: str, contexts: list[str]) -> float:
        """Measure how many retrieved contexts are relevant to the question."""
        if not contexts:
            return 0.0
        
        # Calculate similarity between question and each context
        question_emb = self.embedder.encode([question])
        context_embs = self.embedder.encode(contexts)
        
        similarities = cosine_similarity(question_emb, context_embs)[0]
        
        # Contexts with similarity > threshold are considered relevant
        threshold = 0.3
        relevant_count = sum(1 for sim in similarities if sim > threshold)
        
        return relevant_count / len(contexts)
    
    def _evaluate_context_recall(self, contexts: list[str], ground_truth: str) -> float:
        """Measure if contexts cover the ground truth information."""
        if not contexts:
            return 0.0
        
        # Calculate similarity between ground truth and contexts
        gt_emb = self.embedder.encode([ground_truth])
        context_embs = self.embedder.encode(contexts)
        
        similarities = cosine_similarity(gt_emb, context_embs)[0]
        
        # Return max similarity (best coverage)
        return float(np.max(similarities))
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        emb1 = self.embedder.encode([text1])
        emb2 = self.embedder.encode([text2])
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(max(0.0, min(1.0, similarity)))
    
    @traceable(name="evaluate_retrieval")
    def evaluate_retrieval(
        self,
        retrieved_docs: list[list[str]],
        relevant_docs: list[list[str]],
        k_values: list[int] = [1, 3, 5, 10]
    ) -> dict:
        """Evaluate retrieval with custom metrics."""
        try:
            metrics = {}
            
            for k in k_values:
                precision_k = self._precision_at_k(retrieved_docs, relevant_docs, k)
                recall_k = self._recall_at_k(retrieved_docs, relevant_docs, k)
                hit_k = self._hit_at_k(retrieved_docs, relevant_docs, k)
                
                metrics[f"precision@{k}"] = precision_k
                metrics[f"recall@{k}"] = recall_k
                metrics[f"hit@{k}"] = hit_k
            
            # MRR and NDCG
            metrics["mrr"] = self._mean_reciprocal_rank(retrieved_docs, relevant_docs)
            metrics["ndcg@10"] = self._ndcg_at_k(retrieved_docs, relevant_docs, 10)
            
            logger.info(f"Retrieval Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Retrieval evaluation error: {e}")
            raise
    
    def _precision_at_k(self, retrieved: list[list[str]], relevant: list[list[str]], k: int) -> float:
        """Calculate Precision@k."""
        precisions = []
        
        for ret, rel in zip(retrieved, relevant):
            ret_k = ret[:k]
            relevant_set = set(rel)
            
            if len(ret_k) == 0:
                precisions.append(0.0)
            else:
                precision = len([doc for doc in ret_k if doc in relevant_set]) / len(ret_k)
                precisions.append(precision)
        
        return float(np.mean(precisions))
    
    def _recall_at_k(self, retrieved: list[list[str]], relevant: list[list[str]], k: int) -> float:
        """Calculate Recall@k."""
        recalls = []
        
        for ret, rel in zip(retrieved, relevant):
            ret_k = ret[:k]
            relevant_set = set(rel)
            
            if len(relevant_set) == 0:
                recalls.append(0.0)
            else:
                recall = len([doc for doc in ret_k if doc in relevant_set]) / len(relevant_set)
                recalls.append(recall)
        
        return float(np.mean(recalls))
    
    def _hit_at_k(self, retrieved: list[list[str]], relevant: list[list[str]], k: int) -> float:
        """Calculate Hit@k (at least one relevant doc in top k)."""
        hits = []
        
        for ret, rel in zip(retrieved, relevant):
            ret_k = ret[:k]
            relevant_set = set(rel)
            
            hit = 1.0 if any(doc in relevant_set for doc in ret_k) else 0.0
            hits.append(hit)
        
        return float(np.mean(hits))
    
    def _mean_reciprocal_rank(self, retrieved: list[list[str]], relevant: list[list[str]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        for ret, rel in zip(retrieved, relevant):
            relevant_set = set(rel)
            
            for i, doc in enumerate(ret):
                if doc in relevant_set:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return float(np.mean(reciprocal_ranks))
    
    def _ndcg_at_k(self, retrieved: list[list[str]], relevant: list[list[str]], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@k."""
        ndcgs = []
        
        for ret, rel in zip(retrieved, relevant):
            ret_k = ret[:k]
            relevant_set = set(rel)
            
            # DCG
            dcg = sum(
                (1.0 if doc in relevant_set else 0.0) / np.log2(i + 2)
                for i, doc in enumerate(ret_k)
            )
            
            # IDCG
            ideal_relevance = [1.0] * min(len(relevant_set), k)
            idcg = sum(
                rel / np.log2(i + 2)
                for i, rel in enumerate(ideal_relevance)
            )
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        
        return float(np.mean(ndcgs))