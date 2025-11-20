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
        # Run critic on all steps, stopping at first error
        critiques = critic.critique_all_steps(problem, solution_steps, stop_at_first_error=True)
        
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
    start_index: int = 0,
    end_index: Optional[int] = None,
    sleep_time: float = 0.5,
    run_name: str = "default_run"
) -> Dict:
    """
    Run evaluation on ProcessBench math split.
    
    Args:
        start_index: Starting index (inclusive) for evaluation
        end_index: Ending index (exclusive) for evaluation (None for all remaining)
        sleep_time: Time to sleep between API calls
        run_name: Name of the run (creates directory results/{run_name}/)
        
    Returns:
        Dictionary with metrics and detailed results
    """
    # Create output directory
    output_dir = os.path.join("results", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading ProcessBench dataset...")
    dataset = load_dataset('Qwen/ProcessBench', split='math')
    
    # Determine the range
    if end_index is None:
        end_index = len(dataset)
    end_index = min(end_index, len(dataset))
    
    if start_index >= end_index:
        print(f"Invalid range: start_index ({start_index}) >= end_index ({end_index})")
        return {"metrics": {}, "results": []}
    
    print(f"Processing samples {start_index} to {end_index-1} (total: {end_index - start_index} samples)")
    
    # Initialize critic agent
    print("Initializing Critic agent...")
    critic = Critic(model_name="gpt-4o-mini-2024-07-18", temperature=0.3)
    
    # Evaluate each problem in the range
    results = []
    skipped = 0
    
    for idx in tqdm(range(start_index, end_index), desc="Evaluating"):
        item = dataset[idx]
        
        # Check if result already exists and was successful
        result_file = os.path.join(output_dir, f"result_{idx}.json")
        if os.path.exists(result_file):
            # Load existing result
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            # Only skip if the result exists and has no error (success=True or error=None)
            if result.get('success', False) and result.get('error') is None:
                results.append(result)
                skipped += 1
                continue
            else:
                # Re-run if there was an error
                print(f"\nRe-running sample {idx} due to previous error: {result.get('error', 'Unknown error')}")
        
        # Run evaluation
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
        
        # Save individual result immediately
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    if skipped > 0:
        print(f"\nSkipped {skipped} existing results")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)
    
    # Save aggregated results
    output_data = {
        "metrics": metrics,
        "results": results,
        "config": {
            "start_index": start_index,
            "end_index": end_index,
            "run_name": run_name,
            "total_samples": end_index - start_index
        }
    }
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - Individual results: result_{{index}}.json")
    print(f"  - Summary: summary.json")
    
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
    # Run evaluation on a range of samples
    # Adjust parameters as needed:
    # - start_index: Starting index (inclusive)
    # - end_index: Ending index (exclusive), None for all remaining
    # - sleep_time: Time to sleep between API calls (adjust based on rate limits)
    # - run_name: Name for this run (creates results/{run_name}/ directory)
    
    # Example: Process first 100 samples
    output_data = run_evaluation(
        start_index=0,
        end_index=100,  # Set to None to process all samples
        sleep_time=0.0005,
        run_name="gpt-4o-mini"
    )
    
    print_metrics(output_data['metrics'])
