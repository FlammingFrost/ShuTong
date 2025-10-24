"""
Test the Agent Pipeline with real math problems.

This test demonstrates the pipeline solving actual math problems with
critique and refinement.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_sys import AgentPipeline


def load_problems():
    """Load test problems from JSON file."""
    problems_file = Path(__file__).parent / "test_problems.json"
    with open(problems_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {p['id']: p for p in data['problems']}


def test_gamma_distribution_sum():
    """
    Test problem 1: Gamma distribution sum property.
    
    Problem: If X_i are independently distributed according to Γ(α_i, b), 
    show that ∑X_i is distributed as Γ(∑α_i, b).
    """
    # Load problems from JSON
    problems = load_problems()
    problem_data = problems['gamma_distribution_sum']
    
    print("=" * 80)
    print(f"TEST 1: {problem_data['title'].upper()}")
    print("=" * 80)
    
    # Initialize pipeline with default settings (max_iterations=3)
    pipeline = AgentPipeline(
        solver_model="gpt-4o",
        critic_model="gpt-4o",
        tracker_dir="./data/tracker"
    )
    
    # Get problem statement from JSON
    problem_1 = problem_data['problem']
    
    print("\nProblem:")
    print(problem_1)
    print("\n" + "=" * 80)
    print("Running pipeline...")
    print("=" * 80 + "\n")
    
    # Run the pipeline
    result = pipeline.run(problem_1)
    
    # Print formatted result
    print(pipeline.format_result(result))
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Initial solution steps: {len(result['solution_steps'])}")
    print(f"Refinement iterations: {result['iteration_count']}")
    print(f"Knowledge points identified: {len(result['knowledge_points'])}")
    print(f"Remaining issues: {len(result['feedbacks'])}")
    
    return result


def test_causal_inference_ols():
    """
    Test problem 2: Causal inference with OLS prediction models.
    
    Problem: Show properties of OLS estimators in treatment effect estimation.
    """
    # Load problems from JSON
    problems = load_problems()
    problem_data = problems['causal_inference_ols']
    
    print("\n\n" + "=" * 80)
    print(f"TEST 2: {problem_data['title'].upper()}")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = AgentPipeline(
        solver_model="gpt-4o",
        critic_model="gpt-4o",
        tracker_dir="./data/tracker"
    )
    
    # Get problem statement from JSON
    problem_2 = problem_data['problem']
    
    print("\nProblem:")
    print(problem_2)
    print("\n" + "=" * 80)
    print("Running pipeline...")
    print("=" * 80 + "\n")
    
    # Run the pipeline
    result = pipeline.run(problem_2)
    
    # Print formatted result
    print(pipeline.format_result(result))
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Initial solution steps: {len(result['solution_steps'])}")
    print(f"Refinement iterations: {result['iteration_count']}")
    print(f"Knowledge points identified: {len(result['knowledge_points'])}")
    print(f"Remaining issues: {len(result['feedbacks'])}")
    
    return result


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AGENT PIPELINE TEST SUITE")
    print("=" * 80)
    print("\nThis test suite will run the agent pipeline on two math problems:")
    print("1. Gamma distribution sum property")
    print("2. Causal inference with OLS prediction models")
    print("\nThe pipeline will:")
    print("- Generate initial solutions")
    print("- Critique each step for logic and calculation errors")
    print("- Refine solutions based on feedback (up to 3 iterations)")
    print("- Track all details in the database")
    print("\n" + "=" * 80)
    
    results = {}
    
    # Test 1: Gamma distribution
    try:
        results['gamma'] = test_gamma_distribution_sum()
    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Causal inference
    try:
        results['ols'] = test_causal_inference_ols()
    except Exception as e:
        print(f"\n❌ Test 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        print(f"\n{test_name.upper()}:")
        print(f"  Iterations: {result['iteration_count']}")
        print(f"  Knowledge points: {len(result['knowledge_points'])}")
        print(f"  Final steps: {len(result['solution_steps'])}")
        if result['feedbacks']:
            print(f"  ⚠️  Remaining issues: {len(result['feedbacks'])}")
        else:
            print(f"  ✅ No remaining issues")
    
    print("\n" + "=" * 80)
    print("All tracked data saved to: ./data/tracker/record.db")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
