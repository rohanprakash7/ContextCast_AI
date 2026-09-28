"""
RAGAS Evaluation Runner
Evaluates RAG system quality using RAGAS metrics
"""

import os
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from chains.qa_chain import answer_question
from retrieval.simple_retriever import create_simple_retriever
from augmentation.prompt_templates import format_docs_for_prompt
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Set environment variables for RAGAS
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


class RAGEvaluator:
    """
    Evaluate RAG system using RAGAS metrics
    
    Measures:
    - Faithfulness: Answer grounded in context (no hallucination)
    - Answer Relevancy: Answer addresses the question
    - Context Precision: Retrieved chunks are relevant
    - Context Recall: All relevant chunks retrieved
    """
    
    def __init__(self, video_id: str):
        """
        Initialize evaluator for a video
        
        Args:
            video_id: YouTube video ID to evaluate
        """
        self.video_id = video_id
        logger.info(f"Initialized RAGEvaluator for video: {video_id}")
    
    def evaluate_questions(
        self,
        test_data: List[Dict[str, str]],
        retriever_type: str = "simple"
    ) -> Dict:
        """
        Evaluate RAG system on test questions
        
        Args:
            test_data: List of dicts with 'question' and 'ground_truth'
            retriever_type: Which retriever to evaluate
        
        Returns:
            RAGAS evaluation results
        
        Example test_data:
        [
            {
                "question": "What is deep learning?",
                "ground_truth": "Deep learning is a machine learning technique..."
            },
            ...
        ]
        """
        logger.info(f"Evaluating {len(test_data)} questions")
        
        # Prepare data for RAGAS
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        # Run RAG pipeline for each question
        for item in test_data:
            question = item['question']
            ground_truth = item['ground_truth']
            
            logger.info(f"Processing: {question[:50]}...")
            
            try:
                # Get retriever
                retriever = create_simple_retriever(
                    video_id=self.video_id,
                    top_k=4
                )
                
                # Retrieve documents
                docs = retriever.invoke(question)
                
                # Format context
                context_list = [doc.page_content for doc in docs]
                
                # Get answer
                result = answer_question(
                    question=question,
                    video_id=self.video_id,
                    retriever_type=retriever_type,
                    include_citations=False,
                    top_k=4
                )
                
                # Collect data
                questions.append(question)
                answers.append(result['answer'])
                contexts.append(context_list)
                ground_truths.append(ground_truth)
                
            except Exception as e:
                logger.error(f"Failed to process question: {e}")
                continue
        
        # Create RAGAS dataset
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })
        
        logger.info("Running RAGAS evaluation...")
        
        # Run RAGAS evaluation
        result = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        
        logger.info("Evaluation complete!")
        
        return result


def run_simple_evaluation(video_id: str = "kY14KfZQ1TI"):
    """
    Run evaluation with sample questions
    
    Quick evaluation for testing/demo purposes
    """
    # Sample test data (3 questions for quick test)
    test_data = [
        {
            "question": "What is deep learning?",
            "ground_truth": "Deep learning is a machine learning approach that uses neural networks with multiple layers to learn hierarchical representations from data."
        },
        {
            "question": "How do neural networks work?",
            "ground_truth": "Neural networks work by passing data through layers of interconnected nodes, adjusting weights through backpropagation to learn patterns."
        },
        {
            "question": "What is the purpose of this course?",
            "ground_truth": "The course teaches deep learning fundamentals with focus on self-driving cars and autonomous systems."
        }
    ]
    
    # Run evaluation
    evaluator = RAGEvaluator(video_id=video_id)
    results = evaluator.evaluate_questions(test_data, retriever_type="simple")
    
    # Print results
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Results")
    print("=" * 60)
    print(f"\nVideo ID: {video_id}")
    print(f"Questions evaluated: {len(test_data)}")
    print(f"Retriever type: simple")
    print("\nMetrics:")
    print(f"  Faithfulness:        {results['faithfulness']:.3f}")
    print(f"  Answer Relevancy:    {results['answer_relevancy']:.3f}")
    print(f"  Context Precision:   {results['context_precision']:.3f}")
    print(f"  Context Recall:      {results['context_recall']:.3f}")
    print("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    # Run simple evaluation
    results = run_simple_evaluation()
    
    # Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(dict(results), f, indent=2)
    
    print("\nResults saved to: evaluation_results.json")