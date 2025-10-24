"""
Example usage of the Agent Pipeline.

This demonstrates how to use the integrated pipeline to solve math problems
with automatic critique and refinement.
"""

from agent_sys.pipeline import AgentPipeline


def example_basic_usage():
    """Basic usage example: solve a math problem with default settings."""
    # Initialize the pipeline (default max_iterations is 3)
    pipeline = AgentPipeline(
        solver_model="gpt-4o",
        critic_model="gpt-4o",
        tracker_dir="./data/tracker"
    )
    
    # Define a math problem
    math_problem = """
    Prove that for any continuous random variable X with cumulative distribution 
    function F(x), the random variable Y = F(X) follows a uniform distribution on [0,1].
    """
    
    # Run the pipeline
    result = pipeline.run(math_problem)
    
    # Print formatted result
    print(pipeline.format_result(result))
    
    # Or access specific parts
    print("\n" + "="*80)
    print("FINAL SOLUTION TEXT:")
    print("="*80)
    print(pipeline.get_solution_text(result))
    
    print("\n" + "="*80)
    print("KNOWLEDGE POINTS:")
    print("="*80)
    for kp in pipeline.get_knowledge_points(result):
        print(f"- {kp}")


def example_with_multiple_iterations():
    """Example with custom number of refinement iterations."""
    pipeline = AgentPipeline(max_iterations=5)  # Allow up to 5 refinements
    
    math_problem = """
    Find the derivative of f(x) = x^3 * sin(x) using the product rule.
    """
    
    result = pipeline.run(math_problem)
    
    print(f"Performed {result['iteration_count']} refinement iterations")
    print(f"Final solution has {len(result['solution_steps'])} steps")
    print(f"Identified {len(result['knowledge_points'])} knowledge points")


def example_custom_configuration():
    """Example with custom model configuration."""
    pipeline = AgentPipeline(
        solver_model="gpt-4o",
        critic_model="gpt-4o",
        solver_temperature=0.8,  # Higher creativity for solver
        critic_temperature=0.1,  # Lower temperature for more consistent critique
        max_iterations=3,
        tracker_dir="./data/tracker/custom"
    )
    
    math_problem = """
    Solve the differential equation: dy/dx + 2y = e^(-x), with initial condition y(0) = 1.
    """
    
    result = pipeline.run(math_problem, max_iterations=2)  # Override default
    
    print(pipeline.format_result(result))


def example_accessing_critiques():
    """Example showing how to access detailed critiques."""
    pipeline = AgentPipeline()
    
    math_problem = """
    Calculate the integral of x^2 * e^x dx using integration by parts.
    """
    
    result = pipeline.run(math_problem)
    
    # Access individual critiques
    for critique in result['all_critiques']:
        step_num = critique['step_number']
        print(f"\n--- Step {step_num}: {critique['step_description']} ---")
        print(f"Logic Correct: {critique['is_logically_correct']}")
        print(f"Calculation Correct: {critique['is_calculation_correct']}")
        print(f"Knowledge Points: {', '.join(critique['knowledge_points'])}")


if __name__ == "__main__":
    # Run examples
    print("="*80)
    print("BASIC USAGE EXAMPLE")
    print("="*80)
    example_basic_usage()
