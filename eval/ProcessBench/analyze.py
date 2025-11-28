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
    - Correct match: (both positive) OR (both are -1)
      where positive means != -1
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with match metrics
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
            "correct_match_count": 0,
            "correct_match_rate": 0.0,
            "details": {
                "both_positive_correct": 0,
                "both_negative_correct": 0,
                "exact_positive_matches": 0,
                "exact_negative_matches": 0
            }
        }
    
    exact_match_count = 0
    correct_match_count = 0
    both_positive_correct = 0
    both_negative_correct = 0
    exact_positive_matches = 0
    exact_negative_matches = 0
    
    for result in valid_results:
        predicted = result['predicted_error_step']
        ground_truth = result['ground_truth_label']
        
        # Check exact match
        if predicted == ground_truth:
            exact_match_count += 1
            
            # Track whether it's positive or negative match
            if predicted == -1:
                exact_negative_matches += 1
            else:
                exact_positive_matches += 1
        
        # Check correct match (both positive or both negative)
        predicted_is_positive = predicted != -1
        ground_truth_is_positive = ground_truth != -1
        
        if predicted_is_positive == ground_truth_is_positive:
            correct_match_count += 1
            
            # Track type of correct match
            if predicted_is_positive:
                both_positive_correct += 1
            else:
                both_negative_correct += 1
    
    num_valid = len(valid_results)
    
    return {
        "total_samples": len(results),
        "valid_samples": num_valid,
        "failed_samples": len(results) - num_valid,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / num_valid if num_valid > 0 else 0.0,
        "correct_match_count": correct_match_count,
        "correct_match_rate": correct_match_count / num_valid if num_valid > 0 else 0.0,
        "details": {
            "both_positive_correct": both_positive_correct,
            "both_negative_correct": both_negative_correct,
            "exact_positive_matches": exact_positive_matches,
            "exact_negative_matches": exact_negative_matches
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
    print(f"  - Exact positive matches (step number matches): {metrics['details']['exact_positive_matches']}")
    print(f"  - Exact negative matches (both -1):            {metrics['details']['exact_negative_matches']}")
    print()
    print("-" * 80)
    print("CORRECT MATCH METRIC")
    print("-" * 80)
    print(f"Correct Match Count: {metrics['correct_match_count']} / {metrics['valid_samples']}")
    print(f"Correct Match Rate:  {metrics['correct_match_rate']:.4f} ({metrics['correct_match_rate']*100:.2f}%)")
    print(f"  - Both positive (error detected):  {metrics['details']['both_positive_correct']}")
    print(f"  - Both negative (no error):        {metrics['details']['both_negative_correct']}")
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


if __name__ == "__main__":
    import sys
    
    # Get results base directory from command line or use default
    results_base_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    # Analyze all runs
    all_analyses = analyze_all_runs(results_base_dir)
    
    # Save summary
    if all_analyses:
        output_file = os.path.join(results_base_dir, "analysis_summary.json")
        save_analysis_summary(all_analyses, output_file)
