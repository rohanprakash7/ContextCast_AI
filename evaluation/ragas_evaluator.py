"""
RAGAS Evaluation Runner - FAST VERSION
Simple RAG evaluation using RAGAS framework
Compatible with RAGAS 0.4.3+

OPTIMIZED FOR SPEED:
- 3 questions (instead of 10)
- 3 metrics (skips Answer Relevancy due to embeddings issue)
- Estimated time: 3-5 minutes
- Estimated cost: $0.10-0.20
"""

import sys
import os
from typing import List, Dict
import pandas as pd

# Add parent directory to Python path (fixes ModuleNotFoundError)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import Dataset
from ragas import evaluate

# No need to import specific metrics - let RAGAS handle it

from chains.qa_chain import answer_question
from retrieval import create_retriever
from evaluation.test_dataset import get_test_dataset
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


def run_rag_and_collect_data(test_dataset: List[Dict], retriever_type: str = "simple") -> Dict:
    """
    Run RAG system on test questions and collect data for RAGAS
    
    Args:
        test_dataset: List of test cases
        retriever_type: Which retriever to use
    
    Returns:
        Dictionary with questions, answers, contexts, ground_truths for RAGAS
    """
    logger.info(f"Running RAG on {len(test_dataset)} test cases")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for i, test_case in enumerate(test_dataset, 1):
        question = test_case["question"]
        video_id = test_case["video_id"]
        
        logger.info(f"Processing {i}/{len(test_dataset)}: {question[:60]}...")
        
        try:
            # Run RAG pipeline - get answer WITHOUT citations for cleaner evaluation
            result = answer_question(
                question=question,
                video_id=video_id,
                retriever_type=retriever_type,
                include_citations=False,  # Don't include citations for RAGAS
                top_k=4
            )
            
            # Get retrieved contexts using the same retriever_type being evaluated
            retriever = create_retriever(video_id=video_id, retriever_type=retriever_type, top_k=4)
            docs = retriever.invoke(question)
            context_list = [doc.page_content for doc in docs]
            
            # Collect data
            questions.append(question)
            answers.append(result["answer"])
            contexts.append(context_list)
            
            # Add ground truth if available
            if test_case.get("ground_truth"):
                ground_truths.append(test_case["ground_truth"])
            else:
                ground_truths.append("")  # Empty if no ground truth
            
            logger.info(f"✓ Completed: {question[:50]}...")
        
        except Exception as e:
            logger.error(f"Failed on question: {question} - {e}")
            # Skip failed questions
            continue
    
    logger.info(f"Successfully processed {len(questions)}/{len(test_dataset)} questions")
    
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }


def evaluate_rag_system(video_id: str = "kY14KfZQ1TI", retriever_type: str = "simple") -> Dict:
    """
    Evaluate RAG system using RAGAS (FAST VERSION)
    
    Uses 3 metrics (skips Answer Relevancy):
    1. Faithfulness: Answer based on context (no hallucination)
    2. Context Precision: Retrieved chunks are relevant
    3. Context Recall: Retrieved all relevant chunks (needs ground truth)
    
    Args:
        video_id: Video to evaluate
        retriever_type: Retrieval strategy to test
    
    Returns:
        Dictionary with evaluation scores
    
    Example:
        >>> scores = evaluate_rag_system(video_id="kY14KfZQ1TI")
        >>> print(f"Faithfulness: {scores['faithfulness']:.2f}")
    """
    logger.info("=" * 80)
    logger.info("STARTING FAST RAGAS EVALUATION (3 Questions, 3 Metrics)")
    logger.info("=" * 80)
    
    # Step 1: Get test dataset
    logger.info("\nStep 1: Loading test dataset...")
    test_dataset = get_test_dataset(video_id=video_id)
    logger.info(f"Loaded {len(test_dataset)} test cases")
    logger.info(f"Estimated time: 3-5 minutes")
    
    # Step 2: Run RAG and collect data
    logger.info("\nStep 2: Running RAG pipeline on test cases...")
    data = run_rag_and_collect_data(test_dataset, retriever_type=retriever_type)
    
    if len(data['question']) == 0:
        logger.error("No questions were successfully processed!")
        return {
            "error": "All questions failed to process",
            "num_questions": 0
        }
    
    logger.info(f"Collected data for {len(data['question'])} questions")
    
    # Step 3: Create RAGAS dataset
    logger.info("\nStep 3: Creating RAGAS dataset...")
    ragas_dataset = Dataset.from_dict(data)
    logger.info("Dataset created")
    
    # Step 4: Run RAGAS evaluation
    logger.info("\nStep 4: Running RAGAS evaluation...")
    logger.info("⏳ This will take 3-5 minutes (RAGAS uses LLM for evaluation)")
    logger.info("   RAGAS is calling OpenAI API to evaluate each answer...")
    
    try:
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI

        # Explicitly wire up LLM + embeddings so answer_relevancy doesn't return NaN
        llm = ChatOpenAI(model=settings.OPENAI_CHAT_MODEL, api_key=settings.OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)

        for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
            metric.llm = llm
        answer_relevancy.embeddings = embeddings

        result = evaluate(ragas_dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        
        # Step 5: Extract scores from EvaluationResult object
        # Convert to pandas and get only numeric metric columns
        df = result.to_pandas()
        
        # Get only the metric columns (numeric scores)
        metric_columns = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        available_metrics = [col for col in metric_columns if col in df.columns]
        
        # Calculate mean for numeric metrics only
        result_dict = df[available_metrics].mean().to_dict()
        
        scores = {
            "faithfulness": result_dict.get("faithfulness", 0.0),
            "answer_relevancy": result_dict.get("answer_relevancy", None),
            "context_precision": result_dict.get("context_precision", 0.0),
            "context_recall": result_dict.get("context_recall", 0.0),
            "num_questions": len(data["question"]),
            "retriever_type": retriever_type,
            "video_id": video_id
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("RAGAS EVALUATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nResults for video: {video_id}")
        logger.info(f"Retriever type: {retriever_type}")
        logger.info(f"Questions evaluated: {scores['num_questions']}")
        logger.info("\nMetrics:")
        logger.info(f"  Faithfulness:       {scores['faithfulness']:.3f}  (no hallucination)")
        if scores['answer_relevancy'] is not None and not pd.isna(scores['answer_relevancy']):
            logger.info(f"  Answer Relevancy:   {scores['answer_relevancy']:.3f}  (addresses question)")
        else:
            logger.info(f"  Answer Relevancy:   N/A (embeddings issue)")
        logger.info(f"  Context Precision:  {scores['context_precision']:.3f}  (relevant chunks)")
        logger.info(f"  Context Recall:     {scores['context_recall']:.3f}  (complete retrieval)")
        logger.info("\nScore Interpretation:")
        logger.info("  0.8+ = Excellent")
        logger.info("  0.6-0.8 = Good")
        logger.info("  0.4-0.6 = Needs improvement")
        logger.info("  <0.4 = Significant issues")
        logger.info("\n" + "=" * 80)
        
        return scores
    
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        logger.error("This might be due to OpenAI API issues or data format problems")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    """
    Run fast evaluation from command line
    """
    print("\n" + "=" * 80)
    print("FAST RAGAS RAG EVALUATION - RAGAS 0.4.3")
    print("=" * 80)
    print("\nOptimized for speed and cost:")
    print("  • 3 questions (instead of 10)")
    print("  • 3 metrics (Faithfulness, Context Precision, Context Recall)")
    print("  • Estimated time: 3-5 minutes")
    print("  • Estimated cost: $0.10-0.20")
    print("\nThis will:")
    print("1. Load 3 test questions from test_dataset.py")
    print("2. Run your RAG system on each question")
    print("3. Evaluate with RAGAS metrics using OpenAI")
    print("4. Display scores\n")
    
    # Run evaluation
    try:
        scores = evaluate_rag_system(
            video_id="kY14KfZQ1TI",
            retriever_type="simple"
        )
        
        # Save results
        import json
        import math
        output_file = "evaluation_results_fast.json"
        clean_scores = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in scores.items()}
        with open(output_file, 'w') as f:
            json.dump(clean_scores, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your OpenAI API key is set in config/settings.py")
        print("2. Verify Pinecone connection is working")
        print("3. Make sure you have indexed the video transcript")
        import sys
        sys.exit(1)