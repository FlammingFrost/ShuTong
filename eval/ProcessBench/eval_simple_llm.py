"""
Evaluation pipeline for ProcessBench using a simple LLM approach (non-agentic).

This script evaluates a single powerful LLM call (GPT-4.5-1-2025-11-13) 
to identify the first error in a solution, comparing against the agentic approach.
"""

import json
import time
import re
from typing import List, Dict, Optional
from datasets import load_dataset
from tqdm import tqdm
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def build_prompt(problem: str, steps: List[str]) -> str:
    """
    Build a single prompt for the LLM to identify the first error.
    
    Args:
        problem: The math problem
        steps: List of solution steps
        
    Returns:
        Formatted prompt string
    """
    steps_text = ""
    for i, step in enumerate(steps):
        steps_text += f"\n### Step {i+1}\n{step}\n"
    
    prompt = f"""You are a mathematical critic agent. Your task is to analyze each step of a solution sequentially and identify the FIRST step that contains an error (if any).

**Original Problem:**
{problem}

**Solution Steps:**
{steps_text}

**Instructions:**
For each step in order:
1. Evaluate the logical correctness of the reasoning
2. Verify any calculations for accuracy
3. Identify relevant knowledge points (mathematical concepts, theorems, formulas)
4. Stop at the FIRST step with an error

**Output Format:**
Provide your critique in the following structured format. Be concise - only provide detailed feedback for incorrect steps.

For EACH step analyzed (up to and including the first error), output:

===STEP_[N]===

===LOGICAL_CORRECTNESS===
[TRUE or FALSE]

===LOGICAL_FEEDBACK===
[If FALSE: Provide detailed explanation of the logical error. If TRUE: Write "Correct" or leave empty.]

===CALCULATION_CORRECTNESS===
[TRUE or FALSE]

===CALCULATION_FEEDBACK===
[If FALSE: Provide detailed explanation of the calculation error. If TRUE: Write "Correct" or leave empty.]

===KNOWLEDGE_POINTS===
[Comma-separated list of knowledge point tags, e.g., induction, gamma_distribution, transformation_of_variables]

===END_STEP_[N]===

After analyzing all steps (or stopping at first error), provide:

===FIRST_ERROR_STEP===
[Step number (1-based) of the first error, or -1 if no error found]

Focus on being thorough but concise. Match the output verbosity of an agentic critic system."""
    
    return prompt


def parse_llm_response(response_text: str, num_steps: int) -> Dict:
    """
    Parse the LLM response to extract the error step and analysis.
    
    Args:
        response_text: The LLM's response text
        num_steps: Total number of steps in the solution
        
    Returns:
        Dictionary with parsed information
    """
    # Extract first error step
    error_step_pattern = r"===\s*FIRST_ERROR_STEP\s*===\s*\n\s*(-?\d+)"
    error_step_match = re.search(error_step_pattern, response_text, re.IGNORECASE)
    
    if error_step_match:
        error_step = int(error_step_match.group(1))
        # Convert to 0-based index, or keep -1 for no error
        predicted_error_step = error_step - 1 if error_step > 0 else -1
    else:
        # Fallback: scan each step's critique to find first error
        predicted_error_step = -1
        for step_num in range(1, num_steps + 1):
            # Check if this step has logical or calculation error
            step_pattern = rf"===\s*STEP_{step_num}\s*===.*?===\s*END_STEP_{step_num}\s*==="
            step_match = re.search(step_pattern, response_text, re.DOTALL | re.IGNORECASE)
            
            if step_match:
                step_content = step_match.group(0)
                
                # Check logical correctness
                logic_pattern = r"===\s*LOGICAL_CORRECTNESS\s*===\s*\n\s*(\w+)"
                logic_match = re.search(logic_pattern, step_content, re.IGNORECASE)
                is_logic_correct = logic_match.group(1).strip().upper() == "TRUE" if logic_match else True
                
                # Check calculation correctness
                calc_pattern = r"===\s*CALCULATION_CORRECTNESS\s*===\s*\n\s*(\w+)"
                calc_match = re.search(calc_pattern, step_content, re.IGNORECASE)
                is_calc_correct = calc_match.group(1).strip().upper() == "TRUE" if calc_match else True
                
                # If either is incorrect, this is the first error
                if not is_logic_correct or not is_calc_correct:
                    predicted_error_step = step_num - 1  # Convert to 0-based
                    break
    
    # Determine error type based on the first error step's critique
    error_type = "no_error"
    explanation = ""
    
    if predicted_error_step >= 0:
        step_num = predicted_error_step + 1  # Convert back to 1-based
        step_pattern = rf"===\s*STEP_{step_num}\s*===.*?===\s*END_STEP_{step_num}\s*==="
        step_match = re.search(step_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if step_match:
            step_content = step_match.group(0)
            
            # Check what type of error
            logic_pattern = r"===\s*LOGICAL_CORRECTNESS\s*===\s*\n\s*(\w+)"
            logic_match = re.search(logic_pattern, step_content, re.IGNORECASE)
            is_logic_correct = logic_match.group(1).strip().upper() == "TRUE" if logic_match else True
            
            calc_pattern = r"===\s*CALCULATION_CORRECTNESS\s*===\s*\n\s*(\w+)"
            calc_match = re.search(calc_pattern, step_content, re.IGNORECASE)
            is_calc_correct = calc_match.group(1).strip().upper() == "TRUE" if calc_match else True
            
            if not is_logic_correct and not is_calc_correct:
                error_type = "both"
            elif not is_logic_correct:
                error_type = "logical_error"
            elif not is_calc_correct:
                error_type = "calculation_error"
            
            # Extract feedback for explanation
            logic_feedback_pattern = r"===\s*LOGICAL_FEEDBACK\s*===\s*\n(.*?)(?=\n===)"
            logic_feedback_match = re.search(logic_feedback_pattern, step_content, re.DOTALL | re.IGNORECASE)
            logic_feedback = logic_feedback_match.group(1).strip() if logic_feedback_match else ""
            
            calc_feedback_pattern = r"===\s*CALCULATION_FEEDBACK\s*===\s*\n(.*?)(?=\n===)"
            calc_feedback_match = re.search(calc_feedback_pattern, step_content, re.DOTALL | re.IGNORECASE)
            calc_feedback = calc_feedback_match.group(1).strip() if calc_feedback_match else ""
            
            # Combine feedbacks for explanation
            explanation_parts = []
            if logic_feedback and logic_feedback.lower() != "correct":
                explanation_parts.append(f"Logical error: {logic_feedback}")
            if calc_feedback and calc_feedback.lower() != "correct":
                explanation_parts.append(f"Calculation error: {calc_feedback}")
            explanation = " | ".join(explanation_parts) if explanation_parts else f"Error in step {step_num}"
    else:
        explanation = "All steps are correct"
    
    # Extract all step analyses for summary
    step_analysis = ""
    for step_num in range(1, num_steps + 1):
        step_pattern = rf"===\s*STEP_{step_num}\s*===.*?===\s*END_STEP_{step_num}\s*==="
        step_match = re.search(step_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if step_match:
            step_analysis += f"Step {step_num} analyzed\n"
        if predicted_error_step == step_num - 1:
            break  # Stop after first error
    
    return {
        "predicted_error_step": predicted_error_step,
        "error_type": error_type,
        "explanation": explanation,
        "step_analysis": step_analysis.strip(),
        "raw_response": response_text
    }


def evaluate_single_problem_simple(
    llm: ChatOpenAI,
    problem: str,
    steps: List[str],
    label: int,
    sleep_time: float = 0.1
) -> Dict:
    """
    Evaluate a single problem using simple LLM approach.
    
    Args:
        llm: ChatOpenAI instance
        problem: The math problem
        steps: List of solution steps
        label: Ground truth label (step index with error, or -1 for no error)
        sleep_time: Time to sleep after API calls to avoid rate limits
        
    Returns:
        Dictionary with evaluation results including token usage
    """
    try:
        # Build the prompt - use similar system prompt as critic agent
        system_prompt = "You are a mathematical critic agent that evaluates solution steps for logical correctness, calculation accuracy, and identifies knowledge points."
        user_prompt = build_prompt(problem, steps)
        
        # Make single LLM call
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = response.usage_metadata.get('input_tokens', 0)
            output_tokens = response.usage_metadata.get('output_tokens', 0)
        elif hasattr(response, 'response_metadata') and response.response_metadata:
            token_usage = response.response_metadata.get('token_usage', {})
            input_tokens = token_usage.get('prompt_tokens', 0)
            output_tokens = token_usage.get('completion_tokens', 0)
        
        # Parse the response
        parsed = parse_llm_response(response.content, len(steps))
        
        # Sleep to avoid rate limits
        time.sleep(sleep_time)
        
        result = {
            "predicted_error_step": parsed["predicted_error_step"],
            "ground_truth_label": label,
            "num_steps": len(steps),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error_type": parsed["error_type"],
            "explanation": parsed["explanation"],
            "step_analysis": parsed["step_analysis"],
            "success": True,
            "error": None
        }
        
    except Exception as e:
        result = {
            "predicted_error_step": -1,
            "ground_truth_label": label,
            "num_steps": len(steps),
            "input_tokens": 0,
            "output_tokens": 0,
            "error_type": "unknown",
            "explanation": "",
            "step_analysis": "",
            "success": False,
            "error": str(e)
        }
    
    return result


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate accuracy, precision, recall, and F1 score.
    
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
            "failed_samples": len(results),
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_total_tokens": 0.0
        }
    
    tp = 0  # True Positives
    fp = 0  # False Positives
    tn = 0  # True Negatives
    fn = 0  # False Negatives
    
    exact_matches = 0
    
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
    
    # Calculate token statistics
    total_input_tokens = sum(r.get('input_tokens', 0) for r in valid_results)
    total_output_tokens = sum(r.get('output_tokens', 0) for r in valid_results)
    total_tokens = total_input_tokens + total_output_tokens
    
    avg_input_tokens = total_input_tokens / len(valid_results) if valid_results else 0.0
    avg_output_tokens = total_output_tokens / len(valid_results) if valid_results else 0.0
    avg_total_tokens = total_tokens / len(valid_results) if valid_results else 0.0
    
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
        "failed_samples": len(results) - len(valid_results),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_total_tokens": avg_total_tokens
    }


def run_evaluation(
    start_index: int = 0,
    end_index: Optional[int] = None,
    sleep_time: float = 0.5,
    run_name: str = "gpt-5.1-2025-11-13"
) -> Dict:
    """
    Run evaluation using simple LLM approach.
    
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
    
    # Initialize LLM (GPT-5.1)
    print("Initializing GPT-5.1 model...")
    llm = ChatOpenAI(model="gpt-5.1-2025-11-13", temperature=0.3)
    
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
            
            # Only skip if the result exists and has no error
            if result.get('success', False) and result.get('error') is None:
                results.append(result)
                skipped += 1
                continue
            else:
                # Re-run if there was an error
                print(f"\nRe-running sample {idx} due to previous error: {result.get('error', 'Unknown error')}")
        
        # Run evaluation
        result = evaluate_single_problem_simple(
            llm=llm,
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
            "model": "gpt-5.1-2025-11-13",
            "approach": "simple_llm",
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
    print("EVALUATION METRICS (Simple LLM Approach)")
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
    print()
    print("Token Usage Statistics:")
    print(f"  Total Input Tokens:  {metrics['total_input_tokens']:,}")
    print(f"  Total Output Tokens: {metrics['total_output_tokens']:,}")
    print(f"  Total Tokens:        {metrics['total_tokens']:,}")
    print(f"  Avg Input Tokens:    {metrics['avg_input_tokens']:.1f}")
    print(f"  Avg Output Tokens:   {metrics['avg_output_tokens']:.1f}")
    print(f"  Avg Total Tokens:    {metrics['avg_total_tokens']:.1f}")
    print()
    print("Estimated Cost (GPT-5.1):")
    # GPT-5.1 pricing: $2.50 per 1M input tokens, $10.00 per 1M output tokens
    input_cost = (metrics['total_input_tokens'] / 1_000_000) * 2.50
    output_cost = (metrics['total_output_tokens'] / 1_000_000) * 10.00
    total_cost = input_cost + output_cost
    print(f"  Input Cost:  ${input_cost:.4f}")
    print(f"  Output Cost: ${output_cost:.4f}")
    print(f"  Total Cost:  ${total_cost:.4f}")
    print("="*80)


if __name__ == "__main__":
    # Run evaluation on a range of samples
    # Adjust parameters as needed:
    # - start_index: Starting index (inclusive)
    # - end_index: Ending index (exclusive), None for all remaining
    # - sleep_time: Time to sleep between API calls (adjust based on rate limits)
    # - run_name: Name for this run (creates results/{run_name}/ directory)
    
    # # Example: Process first 100 samples
    # output_data = run_evaluation(
    #     start_index=0,
    #     end_index=100,  # Set to None to process all samples
    #     sleep_time=0.1,
    #     run_name="gpt-5.1-2025-11-13"
    # )
    
    # print_metrics(output_data['metrics'])
    
    output_data = run_evaluation(
        start_index=0,
        end_index=None,  # Set to None to process all samples
        sleep_time=0.1,
        run_name="gpt-5.1-2025-11-13"
    )
    
    print_metrics(output_data['metrics'])
