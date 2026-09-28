"""
RAG System Comparison Evaluator
Tests different retriever configurations and compares results
Saves comparison report for analysis
"""

import sys
import os
from typing import List, Dict
import pandas as pd
import json
from datetime import datetime

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import Dataset
from ragas import evaluate

from chains.qa_chain import answer_question
from retrieval import create_retriever
from evaluation.test_dataset import get_test_dataset
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


def run_rag_and_collect_data(
    test_dataset: List[Dict], 
    retriever_type: str = "simple",
    top_k: int = 4
) -> Dict:
    """
    Run RAG system on test questions and collect data for RAGAS
    
    Args:
        test_dataset: List of test cases
        retriever_type: Which retriever to use (simple, hybrid, rewriting)
        top_k: Number of chunks to retrieve
    
    Returns:
        Dictionary with questions, answers, contexts, ground_truths
    """
    logger.info(f"Running RAG with retriever={retriever_type}, top_k={top_k}")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for i, test_case in enumerate(test_dataset, 1):
        question = test_case["question"]
        video_id = test_case["video_id"]
        
        logger.info(f"Processing {i}/{len(test_dataset)}: {question[:50]}...")
        
        try:
            # Run RAG pipeline
            result = answer_question(
                question=question,
                video_id=video_id,
                retriever_type=retriever_type,
                include_citations=False,
                top_k=top_k
            )
            
            # Get retrieved contexts using the same retriever_type being evaluated
            retriever = create_retriever(video_id=video_id, retriever_type=retriever_type, top_k=top_k)
            docs = retriever.invoke(question)
            context_list = [doc.page_content for doc in docs]
            
            # Collect data
            questions.append(question)
            answers.append(result["answer"])
            contexts.append(context_list)
            ground_truths.append(test_case.get("ground_truth", ""))
            
            logger.info(f"✓ Completed")
        
        except Exception as e:
            logger.error(f"Failed: {e}")
            continue
    
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }


def evaluate_single_configuration(
    video_id: str,
    retriever_type: str,
    top_k: int
) -> Dict:
    """
    Evaluate a single RAG configuration
    
    Args:
        video_id: Video to evaluate
        retriever_type: Type of retriever (simple, hybrid, rewriting)
        top_k: Number of chunks to retrieve
    
    Returns:
        Dictionary with scores and metadata
    """
    logger.info("=" * 80)
    logger.info(f"EVALUATING: retriever={retriever_type}, top_k={top_k}")
    logger.info("=" * 80)
    
    # Get test dataset
    test_dataset = get_test_dataset(video_id=video_id)
    logger.info(f"Loaded {len(test_dataset)} test cases")
    
    # Run RAG and collect data
    data = run_rag_and_collect_data(test_dataset, retriever_type, top_k)
    
    if len(data['question']) == 0:
        logger.error("No questions processed!")
        return {"error": "All questions failed", "config": f"{retriever_type}_k{top_k}"}
    
    # Create RAGAS dataset
    ragas_dataset = Dataset.from_dict(data)
    
    # Run RAGAS evaluation
    logger.info(f"Running RAGAS evaluation...")
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    
    try:
        result = evaluate(ragas_dataset)
        
        # Extract scores
        df = result.to_pandas()
        metric_columns = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        available_metrics = [col for col in metric_columns if col in df.columns]
        result_dict = df[available_metrics].mean().to_dict()
        
        scores = {
            "config_name": f"{retriever_type}_k{top_k}",
            "retriever_type": retriever_type,
            "top_k": top_k,
            "faithfulness": round(result_dict.get("faithfulness", 0.0), 3),
            "answer_relevancy": round(result_dict.get("answer_relevancy", 0.0), 3) if not pd.isna(result_dict.get("answer_relevancy")) else None,
            "context_precision": round(result_dict.get("context_precision", 0.0), 3),
            "context_recall": round(result_dict.get("context_recall", 0.0), 3),
            "num_questions": len(data["question"]),
            "video_id": video_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✓ Evaluation complete!")
        logger.info(f"  Faithfulness: {scores['faithfulness']:.3f}")
        logger.info(f"  Context Precision: {scores['context_precision']:.3f}")
        logger.info(f"  Context Recall: {scores['context_recall']:.3f}")
        
        return scores
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            "config_name": f"{retriever_type}_k{top_k}",
            "error": str(e),
            "retriever_type": retriever_type,
            "top_k": top_k
        }


def run_comparison_study(
    video_id: str = "kY14KfZQ1TI",
    configurations: List[Dict] = None
) -> Dict:
    """
    Run comparison study across multiple configurations
    
    Args:
        video_id: Video to evaluate
        configurations: List of configs to test. Each dict has 'retriever_type' and 'top_k'
    
    Returns:
        Dictionary with all results and comparison analysis
    """
    if configurations is None:
        # Default configurations to test
        configurations = [
            {"retriever_type": "simple", "top_k": 4},  # Baseline
            {"retriever_type": "simple", "top_k": 6},  # Increased retrieval
            {"retriever_type": "hybrid", "top_k": 4},  # Hybrid retriever
            {"retriever_type": "hybrid", "top_k": 6},  # Hybrid + more chunks
        ]
    
    print("\n" + "=" * 80)
    print("RAG SYSTEM COMPARISON STUDY")
    print("=" * 80)
    print(f"\nVideo: {video_id}")
    print(f"Configurations to test: {len(configurations)}")
    print("\nConfigurations:")
    for i, config in enumerate(configurations, 1):
        print(f"  {i}. {config['retriever_type']:10s} with top_k={config['top_k']}")
    print("\nEstimated time: ~10-15 minutes total")
    print("=" * 80)
    
    # Run all evaluations
    all_results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n>>> Testing Configuration {i}/{len(configurations)}")
        print(f">>> Retriever: {config['retriever_type']}, top_k: {config['top_k']}")
        
        result = evaluate_single_configuration(
            video_id=video_id,
            retriever_type=config['retriever_type'],
            top_k=config['top_k']
        )
        
        all_results.append(result)
        print(f">>> Configuration {i} complete!\n")
    
    # Create comparison report
    comparison = create_comparison_report(all_results)
    
    # Save results
    save_comparison_results(comparison)
    
    # Display summary
    display_comparison_summary(comparison)
    
    return comparison


def create_comparison_report(results: List[Dict]) -> Dict:
    """Create structured comparison report"""
    
    # Create DataFrame for easy comparison
    df = pd.DataFrame(results)
    
    # Calculate improvements
    if len(results) > 0:
        baseline = results[0]  # First config is baseline
        
        improvements = []
        for result in results[1:]:
            if "error" not in result:
                improvement = {
                    "config": result['config_name'],
                    "faithfulness_change": result['faithfulness'] - baseline['faithfulness'],
                    "precision_change": result['context_precision'] - baseline['context_precision'],
                    "recall_change": result['context_recall'] - baseline['context_recall']
                }
                improvements.append(improvement)
    
    report = {
        "study_date": datetime.now().isoformat(),
        "baseline": results[0],
        "all_results": results,
        "improvements": improvements if len(results) > 0 else [],
        "best_config": find_best_config(results),
        "recommendations": generate_recommendations(results)
    }
    
    return report


def find_best_config(results: List[Dict]) -> Dict:
    """Find best performing configuration"""
    
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return {}
    
    # Weight metrics: Faithfulness (0.4), Context Recall (0.4), Context Precision (0.2)
    best_score = -1
    best_config = None
    
    for result in valid_results:
        score = (
            result['faithfulness'] * 0.4 +
            result['context_recall'] * 0.4 +
            result['context_precision'] * 0.2
        )
        
        if score > best_score:
            best_score = score
            best_config = result
    
    if best_config:
        best_config['overall_score'] = round(best_score, 3)
    
    return best_config


def generate_recommendations(results: List[Dict]) -> List[str]:
    """Generate actionable recommendations"""
    
    recommendations = []
    
    if len(results) < 2:
        return recommendations
    
    baseline = results[0]
    
    # Check if increasing top_k helped
    for result in results[1:]:
        if "error" in result:
            continue
            
        if result['retriever_type'] == baseline['retriever_type']:
            if result['top_k'] > baseline['top_k']:
                recall_improvement = result['context_recall'] - baseline['context_recall']
                if recall_improvement > 0.1:
                    recommendations.append(
                        f"✓ Increasing top_k from {baseline['top_k']} to {result['top_k']} "
                        f"improved Context Recall by {recall_improvement:.3f}"
                    )
    
    # Check if hybrid helped
    hybrid_results = [r for r in results if r.get('retriever_type') == 'hybrid' and "error" not in r]
    if hybrid_results:
        best_hybrid = max(hybrid_results, key=lambda x: x['context_recall'])
        if best_hybrid['context_recall'] > baseline['context_recall']:
            recommendations.append(
                f"✓ Hybrid retriever improved Context Recall to {best_hybrid['context_recall']:.3f}"
            )
    
    # General recommendations
    best = find_best_config(results)
    if best and best['config_name'] != baseline['config_name']:
        recommendations.append(
            f"✓ RECOMMENDED: Use {best['config_name']} for best overall performance"
        )
    
    return recommendations


def save_comparison_results(comparison: Dict):
    """Save comparison results to JSON file"""
    
    filename = f"evaluation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison results saved to: {filename}")
    print(f"\n✓ Results saved to: {filename}")


def display_comparison_summary(comparison: Dict):
    """Display formatted comparison summary"""
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS SUMMARY")
    print("=" * 80)
    
    # Display all results in table format
    print("\nAll Configurations:")
    print("-" * 80)
    print(f"{'Config':<20} {'Faithfulness':<15} {'Precision':<15} {'Recall':<15}")
    print("-" * 80)
    
    for result in comparison['all_results']:
        if "error" not in result:
            print(f"{result['config_name']:<20} "
                  f"{result['faithfulness']:<15.3f} "
                  f"{result['context_precision']:<15.3f} "
                  f"{result['context_recall']:<15.3f}")
    
    print("-" * 80)
    
    # Display improvements
    if comparison['improvements']:
        print("\nImprovements over Baseline:")
        print("-" * 80)
        baseline = comparison['baseline']
        print(f"Baseline: {baseline['config_name']}")
        
        for improvement in comparison['improvements']:
            print(f"\n{improvement['config']}:")
            print(f"  Faithfulness: {improvement['faithfulness_change']:+.3f}")
            print(f"  Precision:    {improvement['precision_change']:+.3f}")
            print(f"  Recall:       {improvement['recall_change']:+.3f}")
    
    # Display best config
    if comparison['best_config']:
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        best = comparison['best_config']
        print(f"\nConfig: {best['config_name']}")
        print(f"Overall Score: {best['overall_score']:.3f}")
        print(f"\nMetrics:")
        print(f"  Faithfulness:      {best['faithfulness']:.3f}")
        print(f"  Context Precision: {best['context_precision']:.3f}")
        print(f"  Context Recall:    {best['context_recall']:.3f}")
    
    # Display recommendations
    if comparison['recommendations']:
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        for rec in comparison['recommendations']:
            print(f"\n{rec}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    """
    Run comparison study
    """
    
    # Define configurations to test
    configurations = [
        {"retriever_type": "simple", "top_k": 4},   # Baseline (your current config)
        {"retriever_type": "simple", "top_k": 6},   # More chunks
        {"retriever_type": "hybrid", "top_k": 4},   # Hybrid retriever
        {"retriever_type": "hybrid", "top_k": 6},   # Hybrid + more chunks
    ]
    
    # Run comparison
    comparison = run_comparison_study(
        video_id="kY14KfZQ1TI",
        configurations=configurations
    )
    
    print("\n✓ Comparison study complete!")