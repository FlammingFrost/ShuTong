"""
Analyze evaluation results from ProcessBench runs.

This script reads through evaluation results and calculates metrics:
- Exact match: predicted_error_step == ground_truth_label
- Correct match: both are positive (not -1) OR both are -1
"""

import json
import os
from typing import Dict, List
from pathlib import Path


def load_results_from_run(run_name: str, results_base_dir: str = "results") -> List[Dict]:
    """
    Load all result files from a specific run directory.
    
    Args:
        run_name: Name of the run (subdirectory under results/)
        results_base_dir: Base directory containing results
        
    Returns:
        List of result dictionaries
    """
    run_dir = os.path.join(results_base_dir, run_name)
    
    if not os.path.exists(run_dir):
        raise ValueError(f"Run directory not found: {run_dir}")
    
    results = []
    
    # Find all result_*.json files
    for filename in sorted(os.listdir(run_dir)):
        if filename.startswith("result_") and filename.endswith(".json"):
            filepath = os.path.join(run_dir, filename)
            with open(filepath, 'r') as f:
                result = json.load(f)
                results.append(result)
    
    return results


def calculate_match_metrics(results: List[Dict]) -> Dict:
    """
    Calculate exact match and correct match metrics.
    
    Metrics:
    - Exact match: predicted_error_step == ground_truth_label
      Only accuracy is calculated (overall, positive cases, negative cases)
    - Correct match: (both positive) OR (both are -1)
      where positive means != -1
      Precision, recall, accuracy, and F1 are calculated
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with match metrics including token usage
    """
    # Filter successful results only
    valid_results = [r for r in results if r.get('success', False)]
    
    if not valid_results:
        return {
            "total_samples": len(results),
            "valid_samples": 0,
            "failed_samples": len(results),
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "exact_accuracy": 0.0,
            "exact_accuracy_positive": 0.0,
            "exact_accuracy_negative": 0.0,
            "correct_match_count": 0,
            "correct_match_rate": 0.0,
            "correct_precision": 0.0,
            "correct_recall": 0.0,
            "correct_accuracy": 0.0,
            "correct_f1_score": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "details": {
                "exact_correct": 0,
                "exact_total": 0,
                "exact_correct_positive": 0,
                "exact_total_positive": 0,
                "exact_correct_negative": 0,
                "exact_total_negative": 0,
                "correct_tp": 0,
                "correct_fp": 0,
                "correct_tn": 0,
                "correct_fn": 0
            }
        }
    
    # For EXACT MATCH: Only track accuracy
    # Count correct predictions overall, for positive cases, and for negative cases
    exact_correct = 0  # Total correct exact matches
    exact_total = 0  # Total cases
    exact_correct_positive = 0  # Correct among positive (error exists) cases
    exact_total_positive = 0  # Total positive cases
    exact_correct_negative = 0  # Correct among negative (no error) cases
    exact_total_negative = 0  # Total negative cases
    
    # For CORRECT MATCH: Binary classification (correct detection vs incorrect)
    # Positive = correct match (both positive or both negative), Negative = incorrect match
    correct_tp = 0  # Predicted correctly (both error or both no-error) and it is correct
    correct_fp = 0  # Predicted correctly but it's wrong
    correct_tn = 0  # Predicted incorrectly and it is incorrect
    correct_fn = 0  # Predicted incorrectly but should be correct
    
    for result in valid_results:
        predicted = result['predicted_error_step']
        ground_truth = result['ground_truth_label']
        
        # Binary classification: predicting error (not -1) vs no error (-1)
        predicted_is_positive = predicted != -1  # Predicted an error exists
        ground_truth_is_positive = ground_truth != -1  # Ground truth has an error
        
        # EXACT MATCH: Check if predicted_error_step == ground_truth_label
        is_exact_match = (predicted == ground_truth)
        
        exact_total += 1
        if is_exact_match:
            exact_correct += 1
        
        # Track accuracy for positive cases (ground truth has an error)
        if ground_truth_is_positive:
            exact_total_positive += 1
            if is_exact_match:
                exact_correct_positive += 1
        else:
            # Track accuracy for negative cases (ground truth has no error)
            exact_total_negative += 1
            if is_exact_match:
                exact_correct_negative += 1
        
        # CORRECT MATCH: Check if both positive (error detected) OR both negative (no error)
        # Treat this as predicting whether an error exists (any error)
        is_correct_match = (predicted_is_positive == ground_truth_is_positive)
        
        if predicted_is_positive:
            # We predicted an error exists
            if ground_truth_is_positive:
                # Ground truth also has error: True Positive
                correct_tp += 1
            else:
                # Ground truth is -1 (no error): False Positive
                correct_fp += 1
        else:
            # We predicted no error (predicted == -1)
            if not ground_truth_is_positive:
                # Ground truth is also -1: True Negative
                correct_tn += 1
            else:
                # Ground truth has error: False Negative
                correct_fn += 1
    
    num_valid = len(valid_results)
    
    # Calculate token statistics
    total_input_tokens = sum(r.get('input_tokens', 0) for r in valid_results)
    total_output_tokens = sum(r.get('output_tokens', 0) for r in valid_results)
    total_tokens = total_input_tokens + total_output_tokens
    
    avg_input_tokens = total_input_tokens / num_valid if num_valid > 0 else 0.0
    avg_output_tokens = total_output_tokens / num_valid if num_valid > 0 else 0.0
    avg_total_tokens = total_tokens / num_valid if num_valid > 0 else 0.0
    
    # Calculate metrics for EXACT MATCH
    # Only accuracy is calculated: overall, for positive cases, and for negative cases
    exact_accuracy = exact_correct / exact_total if exact_total > 0 else 0.0
    exact_accuracy_positive = exact_correct_positive / exact_total_positive if exact_total_positive > 0 else 0.0
    exact_accuracy_negative = exact_correct_negative / exact_total_negative if exact_total_negative > 0 else 0.0
    
    # Calculate metrics for CORRECT MATCH
    # Precision = TP / (TP + FP) - of all predicted errors, how many were correct (right type)?
    # Recall = TP / (TP + FN) - of all actual errors, how many did we detect?
    # Accuracy = (TP + TN) / (TP + TN + FP + FN) - overall correctness
    correct_precision = correct_tp / (correct_tp + correct_fp) if (correct_tp + correct_fp) > 0 else 0.0
    correct_recall = correct_tp / (correct_tp + correct_fn) if (correct_tp + correct_fn) > 0 else 0.0
    correct_accuracy = (correct_tp + correct_tn) / (correct_tp + correct_tn + correct_fp + correct_fn) if (correct_tp + correct_tn + correct_fp + correct_fn) > 0 else 0.0
    correct_f1 = 2 * (correct_precision * correct_recall) / (correct_precision + correct_recall) if (correct_precision + correct_recall) > 0 else 0.0
    
    return {
        "total_samples": len(results),
        "valid_samples": num_valid,
        "failed_samples": len(results) - num_valid,
        "exact_match_count": exact_correct,
        "exact_match_rate": exact_accuracy,
        "exact_accuracy": exact_accuracy,
        "exact_accuracy_positive": exact_accuracy_positive,
        "exact_accuracy_negative": exact_accuracy_negative,
        "correct_match_count": correct_tp + correct_tn,
        "correct_match_rate": (correct_tp + correct_tn) / (correct_tp + correct_tn + correct_fp + correct_fn) if (correct_tp + correct_tn + correct_fp + correct_fn) > 0 else 0.0,
        # Correct match metrics
        "correct_precision": correct_precision,
        "correct_recall": correct_recall,
        "correct_accuracy": correct_accuracy,
        "correct_f1_score": correct_f1,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_total_tokens": avg_total_tokens,
        "details": {
            "exact_correct": exact_correct,
            "exact_total": exact_total,
            "exact_correct_positive": exact_correct_positive,
            "exact_total_positive": exact_total_positive,
            "exact_correct_negative": exact_correct_negative,
            "exact_total_negative": exact_total_negative,
            "correct_tp": correct_tp,
            "correct_fp": correct_fp,
            "correct_tn": correct_tn,
            "correct_fn": correct_fn
        }
    }


def analyze_run(run_name: str, results_base_dir: str = "results") -> Dict:
    """
    Analyze a single run and calculate match metrics.
    
    Args:
        run_name: Name of the run
        results_base_dir: Base directory containing results
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing run: {run_name}")
    print(f"Loading results from {results_base_dir}/{run_name}/...")
    
    results = load_results_from_run(run_name, results_base_dir)
    print(f"Loaded {len(results)} results")
    
    metrics = calculate_match_metrics(results)
    
    return {
        "run_name": run_name,
        "metrics": metrics
    }


def print_analysis(analysis: Dict):
    """Print analysis results in a readable format."""
    metrics = analysis['metrics']
    
    print("\n" + "="*80)
    print(f"ANALYSIS RESULTS: {analysis['run_name']}")
    print("="*80)
    print(f"Total Samples:  {metrics['total_samples']}")
    print(f"Valid Samples:  {metrics['valid_samples']}")
    print(f"Failed Samples: {metrics['failed_samples']}")
    print()
    print("-" * 80)
    print("EXACT MATCH METRIC")
    print("-" * 80)
    print(f"Exact Match Count: {metrics['exact_match_count']} / {metrics['valid_samples']}")
    print(f"Exact Match Rate:  {metrics['exact_match_rate']:.4f} ({metrics['exact_match_rate']*100:.2f}%)")
    print()
    print("Accuracy Metrics (Exact Match):")
    print(f"  Overall Accuracy:  {metrics['exact_accuracy']:.4f} ({metrics['exact_accuracy']*100:.2f}%)")
    print(f"    - Correct: {metrics['details']['exact_correct']} / {metrics['details']['exact_total']}")
    print(f"  Positive Cases (error exists):  {metrics['exact_accuracy_positive']:.4f} ({metrics['exact_accuracy_positive']*100:.2f}%)")
    print(f"    - Correct: {metrics['details']['exact_correct_positive']} / {metrics['details']['exact_total_positive']}")
    print(f"  Negative Cases (no error):      {metrics['exact_accuracy_negative']:.4f} ({metrics['exact_accuracy_negative']*100:.2f}%)")
    print(f"    - Correct: {metrics['details']['exact_correct_negative']} / {metrics['details']['exact_total_negative']}")
    print()
    print("-" * 80)
    print("CORRECT MATCH METRIC")
    print("-" * 80)
    print(f"Correct Match Count: {metrics['correct_match_count']} / {metrics['valid_samples']}")
    print(f"Correct Match Rate:  {metrics['correct_match_rate']:.4f} ({metrics['correct_match_rate']*100:.2f}%)")
    print()
    print("Binary Classification Metrics (Correct Match):")
    print(f"  Precision: {metrics['correct_precision']:.4f} ({metrics['correct_precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['correct_recall']:.4f} ({metrics['correct_recall']*100:.2f}%)")
    print(f"  Accuracy:  {metrics['correct_accuracy']:.4f} ({metrics['correct_accuracy']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['correct_f1_score']:.4f} ({metrics['correct_f1_score']*100:.2f}%)")
    print(f"  (TP={metrics['details']['correct_tp']}, FP={metrics['details']['correct_fp']}, "
          f"TN={metrics['details']['correct_tn']}, FN={metrics['details']['correct_fn']})")
    print()
    print("-" * 80)
    print("TOKEN USAGE")
    print("-" * 80)
    print(f"Total Input Tokens:  {metrics['total_input_tokens']:,}")
    print(f"Total Output Tokens: {metrics['total_output_tokens']:,}")
    print(f"Total Tokens:        {metrics['total_tokens']:,}")
    print(f"Avg Input Tokens:    {metrics['avg_input_tokens']:.1f}")
    print(f"Avg Output Tokens:   {metrics['avg_output_tokens']:.1f}")
    print(f"Avg Total Tokens:    {metrics['avg_total_tokens']:.1f}")
    print()
    print("Note: Correct match = both positive (any error step) OR both negative (no error)")
    print("="*80)


def analyze_all_runs(results_base_dir: str = "results") -> Dict[str, Dict]:
    """
    Analyze all runs in the results directory.
    
    Args:
        results_base_dir: Base directory containing results
        
    Returns:
        Dictionary mapping run names to their analysis results
    """
    if not os.path.exists(results_base_dir):
        print(f"Results directory not found: {results_base_dir}")
        return {}
    
    run_dirs = [d for d in os.listdir(results_base_dir) 
                if os.path.isdir(os.path.join(results_base_dir, d))]
    
    if not run_dirs:
        print(f"No run directories found in {results_base_dir}")
        return {}
    
    print(f"Found {len(run_dirs)} run(s): {', '.join(run_dirs)}")
    print()
    
    all_analyses = {}
    
    for run_name in run_dirs:
        try:
            analysis = analyze_run(run_name, results_base_dir)
            all_analyses[run_name] = analysis
            print_analysis(analysis)
            print()
        except Exception as e:
            print(f"Error analyzing {run_name}: {e}")
            print()
    
    return all_analyses


def save_analysis_summary(
    all_analyses: Dict[str, Dict],
    output_file: str = "results/analysis_summary.json"
):
    """
    Save analysis summary to a JSON file.
    
    Args:
        all_analyses: Dictionary of all analyses
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        json.dump(all_analyses, f, indent=2)
    
    print(f"Analysis summary saved to: {output_file}")


def compare_runs(run1_name: str, run2_name: str, results_base_dir: str = "results"):
    """
    Compare two evaluation runs side by side.
    
    Args:
        run1_name: Name of first run
        run2_name: Name of second run
        results_base_dir: Base directory containing results
    """
    print("\n" + "="*80)
    print(f"COMPARISON: {run1_name} vs {run2_name}")
    print("="*80)
    
    # Load and analyze both runs
    analysis1 = analyze_run(run1_name, results_base_dir)
    analysis2 = analyze_run(run2_name, results_base_dir)
    
    m1 = analysis1['metrics']
    m2 = analysis2['metrics']
    
    print("\n" + "-"*80)
    print(f"{'Metric':<40} {run1_name:<20} {run2_name:<20}")
    print("-"*80)
    
    # Sample counts
    print(f"{'Valid Samples':<40} {m1['valid_samples']:<20} {m2['valid_samples']:<20}")
    print()
    
    # Match rates
    print(f"{'Exact Match Rate':<40} {m1['exact_match_rate']*100:>18.2f}% {m2['exact_match_rate']*100:>18.2f}%")
    print(f"{'Correct Match Rate':<40} {m1['correct_match_rate']*100:>18.2f}% {m2['correct_match_rate']*100:>18.2f}%")
    print()
    
    # Exact match accuracy metrics
    print(f"{'Exact Match - Overall Accuracy':<40} {m1['exact_accuracy']*100:>18.2f}% {m2['exact_accuracy']*100:>18.2f}%")
    print(f"{'Exact Match - Positive Accuracy':<40} {m1['exact_accuracy_positive']*100:>18.2f}% {m2['exact_accuracy_positive']*100:>18.2f}%")
    print(f"{'Exact Match - Negative Accuracy':<40} {m1['exact_accuracy_negative']*100:>18.2f}% {m2['exact_accuracy_negative']*100:>18.2f}%")
    print()
    
    # Correct match metrics
    print(f"{'Correct Match - Precision':<40} {m1['correct_precision']*100:>18.2f}% {m2['correct_precision']*100:>18.2f}%")
    print(f"{'Correct Match - Recall':<40} {m1['correct_recall']*100:>18.2f}% {m2['correct_recall']*100:>18.2f}%")
    print(f"{'Correct Match - Accuracy':<40} {m1['correct_accuracy']*100:>18.2f}% {m2['correct_accuracy']*100:>18.2f}%")
    print(f"{'Correct Match - F1-Score':<40} {m1['correct_f1_score']*100:>18.2f}% {m2['correct_f1_score']*100:>18.2f}%")
    print()
    
    # Token usage
    print(f"{'Total Tokens':<40} {m1['total_tokens']:>18,} {m2['total_tokens']:>18,}")
    print(f"{'Avg Input Tokens':<40} {m1['avg_input_tokens']:>18.1f} {m2['avg_input_tokens']:>18.1f}")
    print(f"{'Avg Output Tokens':<40} {m1['avg_output_tokens']:>18.1f} {m2['avg_output_tokens']:>18.1f}")
    print(f"{'Avg Total Tokens':<40} {m1['avg_total_tokens']:>18.1f} {m2['avg_total_tokens']:>18.1f}")
    print()
    
    # Cost estimation
    # Detect model based on run name
    if 'gpt-4o-mini' in run1_name.lower():
        cost1_input_rate = 0.150  # per 1M tokens
        cost1_output_rate = 0.600
        model1_name = "GPT-4o-mini"
    elif 'gpt-5-nano' in run1_name.lower() or 'gpt5nano' in run1_name.lower():
        cost1_input_rate = 0.05
        cost1_output_rate = 0.4
        model1_name = "GPT-5-nano"
    elif 'gpt-5-mini' in run1_name.lower() or 'gpt5mini' in run1_name.lower():
        cost1_input_rate = 0.25
        cost1_output_rate = 2.00
        model1_name = "GPT-5-mini"
    elif 'gpt-4.5' in run1_name.lower() or 'gpt-5' in run1_name.lower():
        cost1_input_rate = 2.50
        cost1_output_rate = 10.00
        model1_name = "GPT-4.5-1"
    else:
        cost1_input_rate = 2.50  # default to GPT-4 pricing
        cost1_output_rate = 10.00
        model1_name = "Unknown"
    
    if 'gpt-4o-mini' in run2_name.lower():
        cost2_input_rate = 0.150
        cost2_output_rate = 0.600
        model2_name = "GPT-4o-mini"
    elif 'gpt-5-nano' in run2_name.lower() or 'gpt5nano' in run2_name.lower():
        cost2_input_rate = 0.05
        cost2_output_rate = 0.4
        model2_name = "GPT-5-nano"
    elif 'gpt-5-mini' in run2_name.lower() or 'gpt5mini' in run2_name.lower():
        cost2_input_rate = 0.25
        cost2_output_rate = 2.00
        model2_name = "GPT-5-mini"
    elif 'gpt-5.1' in run2_name.lower():
        cost2_input_rate = 1.25
        cost2_output_rate = 10.00
        model2_name = "GPT-5.1"
    else:
        cost2_input_rate = 2.50
        cost2_output_rate = 10.00
        model2_name = "Unknown"
    
    cost1 = (m1['total_input_tokens'] / 1_000_000 * cost1_input_rate + 
             m1['total_output_tokens'] / 1_000_000 * cost1_output_rate)
    cost2 = (m2['total_input_tokens'] / 1_000_000 * cost2_input_rate + 
             m2['total_output_tokens'] / 1_000_000 * cost2_output_rate)
    
    print(f"{'Estimated Total Cost':<40} ${cost1/m1['valid_samples']*1000:>17.4f} ${cost2/m2['valid_samples']*1000:>17.4f}")
    print(f"{'Cost per Question':<40} ${cost1/m1['valid_samples']:>17.4f} ${cost2/m2['valid_samples']:>17.4f}")
    print(f"{'Model Pricing':<40} {model1_name:<20} {model2_name:<20}")
    print()
    
    # Performance comparison
    print("-"*80)
    print("PERFORMANCE DELTA")
    print("-"*80)
    exact_rate_diff = (m2['exact_match_rate'] - m1['exact_match_rate']) * 100
    correct_rate_diff = (m2['correct_match_rate'] - m1['correct_match_rate']) * 100
    
    # Exact match metric differences
    exact_accuracy_diff = (m2['exact_accuracy'] - m1['exact_accuracy']) * 100
    exact_accuracy_pos_diff = (m2['exact_accuracy_positive'] - m1['exact_accuracy_positive']) * 100
    exact_accuracy_neg_diff = (m2['exact_accuracy_negative'] - m1['exact_accuracy_negative']) * 100
    
    # Correct match metric differences
    correct_precision_diff = (m2['correct_precision'] - m1['correct_precision']) * 100
    correct_recall_diff = (m2['correct_recall'] - m1['correct_recall']) * 100
    correct_accuracy_diff = (m2['correct_accuracy'] - m1['correct_accuracy']) * 100
    correct_f1_diff = (m2['correct_f1_score'] - m1['correct_f1_score']) * 100
    
    cost_diff = cost2 - cost1
    cost_pct = ((cost2 / cost1) - 1) * 100 if cost1 > 0 else 0
    
    print(f"Exact Match Rate Difference:          {exact_rate_diff:+.2f}% ({run2_name} vs {run1_name})")
    print(f"Correct Match Rate Difference:        {correct_rate_diff:+.2f}% ({run2_name} vs {run1_name})")
    print()
    print(f"Exact Match - Overall Accuracy Diff:  {exact_accuracy_diff:+.2f}%")
    print(f"Exact Match - Positive Accuracy Diff: {exact_accuracy_pos_diff:+.2f}%")
    print(f"Exact Match - Negative Accuracy Diff: {exact_accuracy_neg_diff:+.2f}%")
    print()
    print(f"Correct Match - Precision Diff:       {correct_precision_diff:+.2f}%")
    print(f"Correct Match - Recall Diff:          {correct_recall_diff:+.2f}%")
    print(f"Correct Match - Accuracy Diff:        {correct_accuracy_diff:+.2f}%")
    print(f"Correct Match - F1-Score Diff:        {correct_f1_diff:+.2f}%")
    print()
    print(f"Cost Difference:                      ${cost_diff:+.4f} ({cost_pct:+.1f}%)")
    print()
    
    # Cost-effectiveness
    if m2['exact_match_rate'] > 0:
        cost_per_exact1 = cost1 / (m1['exact_match_count']) if m1['exact_match_count'] > 0 else float('inf')
        cost_per_exact2 = cost2 / (m2['exact_match_count']) if m2['exact_match_count'] > 0 else float('inf')
        print(f"Cost per Exact Match:         ${cost_per_exact1:.4f}      ${cost_per_exact2:.4f}")
        
        cost_per_correct1 = cost1 / (m1['correct_match_count']) if m1['correct_match_count'] > 0 else float('inf')
        cost_per_correct2 = cost2 / (m2['correct_match_count']) if m2['correct_match_count'] > 0 else float('inf')
        print(f"Cost per Correct Match:       ${cost_per_correct1:.4f}      ${cost_per_correct2:.4f}")
    
    print("="*80)


def save_analysis_summary(
    all_analyses: Dict[str, Dict],
    output_file: str = "results/analysis_summary.json"
):
    """
    Save analysis summary to a JSON file.
    
    Args:
        all_analyses: Dictionary of all analyses
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        json.dump(all_analyses, f, indent=2)
    
    print(f"Analysis summary saved to: {output_file}")


if __name__ == "__main__":
    import sys
    
    # Check for comparison mode
    if len(sys.argv) >= 3 and sys.argv[1] == "compare":
        # Compare two runs
        run1 = sys.argv[2]
        run2 = sys.argv[3]
        results_base_dir = sys.argv[4] if len(sys.argv) > 4 else "results"
        
        compare_runs(run1, run2, results_base_dir)
    else:
        # Get results base directory from command line or use default
        results_base_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
        
        # Analyze all runs
        all_analyses = analyze_all_runs(results_base_dir)
        
        # Save summary
        if all_analyses:
            output_file = os.path.join(results_base_dir, "analysis_summary.json")
            save_analysis_summary(all_analyses, output_file)
            
            # If there are exactly 2 runs, also show comparison
            if len(all_analyses) == 2:
                print("\n")
                run_names = list(all_analyses.keys())
                compare_runs(run_names[0], run_names[1], results_base_dir)
