"""
Evaluation pipeline for ProcessBench dataset using the Critic Agent.

This script evaluates the critic agent's ability to identify the first error
in a solution by comparing its predictions with ground truth labels.
"""

import json
import time
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from critic_agent.critic import Critic


def convert_to_solution_steps(steps: List[str]) -> List[dict]:
    """
    Convert ProcessBench steps to the format expected by Critic agent.
    
    Args:
        steps: List of step strings from ProcessBench
        
    Returns:
        List of dictionaries with 'description' and 'content' keys
    """
    solution_steps = []
    for i, step in enumerate(steps):
        solution_steps.append({
            "description": f"Step {i+1}",
            "content": step
        })
    return solution_steps


def find_first_error(critiques: List[dict]) -> int:
    """
    Find the first step where the critic detects an error.
    
    Args:
        critiques: List of critique dictionaries from critic agent
        
    Returns:
        Step index (0-based) of first error, or -1 if no error found
    """
    for critique in critiques:
        # Check if either logical or calculation correctness is false
        if not critique['is_logically_correct'] or not critique['is_calculation_correct']:
            # Return 0-based index
            return critique['step_number'] - 1
    
    return -1


def evaluate_single_problem(
    critic: Critic,
    problem: str,
    steps: List[str],
    label: int,
    sleep_time: float = 0.1
) -> Dict:
    """
    Evaluate critic agent on a single problem.
    
    Args:
        critic: Critic agent instance
        problem: The math problem
        steps: List of solution steps
        label: Ground truth label (step index with error, or -1 for no error)
        sleep_time: Time to sleep after API calls to avoid rate limits
        
    Returns:
        Dictionary with evaluation results
    """
    solution_steps = convert_to_solution_steps(steps)
    
    try:
        # Run critic on all steps
        critiques = critic.critique_all_steps(problem, solution_steps)
        
        # Find first error detected by critic
        predicted_error_step = find_first_error(critiques)
        
        # Sleep to avoid rate limits
        time.sleep(sleep_time)
        
        result = {
            "predicted_error_step": predicted_error_step,
            "ground_truth_label": label,
            "num_steps": len(steps),
            "success": True,
            "error": None
        }
        
    except Exception as e:
        result = {
            "predicted_error_step": -1,
            "ground_truth_label": label,
            "num_steps": len(steps),
            "success": False,
            "error": str(e)
        }
    
    return result


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate accuracy, precision, recall, and F1 score.
    
    For this task:
    - Positive class: There IS an error (label != -1)
    - Negative class: There is NO error (label == -1)
    
    Metrics:
    - Accuracy: Correct predictions / Total predictions
    - Precision: True Positives / (True Positives + False Positives)
    - Recall: True Positives / (True Positives + False Negatives)
    - F1: Harmonic mean of precision and recall
    
    Where:
    - True Positive: Predicted error exists AND ground truth has error
    - False Positive: Predicted error exists BUT ground truth has no error
    - True Negative: Predicted no error AND ground truth has no error
    - False Negative: Predicted no error BUT ground truth has error
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with metrics
    """
    # Filter successful results
    valid_results = [r for r in results if r['success']]
    
    if not valid_results:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "total_samples": len(results),
            "valid_samples": 0,
            "failed_samples": len(results)
        }
    
    tp = 0  # True Positives: predicted error and has error
    fp = 0  # False Positives: predicted error but no error
    tn = 0  # True Negatives: predicted no error and no error
    fn = 0  # False Negatives: predicted no error but has error
    
    exact_matches = 0  # Count exact step matches (for additional insight)
    
    for result in valid_results:
        predicted = result['predicted_error_step']
        ground_truth = result['ground_truth_label']
        
        # Check if prediction matches exactly
        if predicted == ground_truth:
            exact_matches += 1
        
        # Binary classification: error exists or not
        predicted_has_error = predicted != -1
        ground_truth_has_error = ground_truth != -1
        
        if predicted_has_error and ground_truth_has_error:
            tp += 1
        elif predicted_has_error and not ground_truth_has_error:
            fp += 1
        elif not predicted_has_error and not ground_truth_has_error:
            tn += 1
        elif not predicted_has_error and ground_truth_has_error:
            fn += 1
    
    # Calculate metrics
    accuracy = (tp + tn) / len(valid_results) if valid_results else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match_rate = exact_matches / len(valid_results) if valid_results else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact_match_rate,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_samples": len(results),
        "valid_samples": len(valid_results),
        "failed_samples": len(results) - len(valid_results)
    }


def run_evaluation(
    max_samples: Optional[int] = None,
    sleep_time: float = 0.5,
    output_file: str = "eval_results.json"
) -> Dict:
    """
    Run evaluation on ProcessBench math split.
    
    Args:
        max_samples: Maximum number of samples to process (None for all)
        sleep_time: Time to sleep between API calls
        output_file: Path to save detailed results
        
    Returns:
        Dictionary with metrics and detailed results
    """
    # Load dataset
    print("Loading ProcessBench dataset...")
    dataset = load_dataset('Qwen/ProcessBench', split='math')
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Processing {len(dataset)} samples...")
    
    # Initialize critic agent
    print("Initializing Critic agent...")
    # critic = Critic(model_name="gpt-4o", temperature=0.3)
    critic = Critic(model_name="gpt-4o-mini-2024-07-18", temperature=0.3)
    
    # Evaluate each problem
    results = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        result = evaluate_single_problem(
            critic=critic,
            problem=item['problem'],
            steps=item['steps'],
            label=item['label'],
            sleep_time=sleep_time
        )
        result['id'] = item['id']
        result['index'] = idx
        results.append(result)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)
    
    # Save detailed results
    output_data = {
        "metrics": metrics,
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    return output_data


def print_metrics(metrics: Dict):
    """Print metrics in a readable format."""
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Valid Samples: {metrics['valid_samples']}")
    print(f"Failed Samples: {metrics['failed_samples']}")
    print()
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.4f} ({metrics['exact_match_rate']*100:.2f}%)")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print("="*80)


if __name__ == "__main__":
    # Run evaluation on all 1000+ samples
    # Adjust sleep_time based on your API rate limits
    # OpenAI typically allows 500 requests/min for standard tier
    # With sleep_time=0.5, we make ~2 requests/second = 120/min (safe)
    
    output_data = run_evaluation(
        max_samples=None,  # Process all samples
        sleep_time=0.0005,     # 0.5 seconds between problems
        output_file="eval_results.json"
    )
    
    print_metrics(output_data['metrics'])
