"""
State definition for the Agent System Pipeline.
"""

from typing import TypedDict, List, Dict, Optional
from solver_agent.state import SolutionStep
from critic_agent.state import StepCritique


class PipelineState(TypedDict):
    """State for the integrated agent pipeline."""
    # Input
    math_problem: str
    
    # Solver outputs
    initial_solution: str
    solution_steps: List[SolutionStep]
    
    # Critic outputs (collections)
    all_critiques: List[StepCritique]
    feedbacks: List[str]
    knowledge_points: List[str]
    
    # Refined solution
    refined_solution: Optional[str]
    refined_steps: Optional[List[SolutionStep]]
    
    # Control flags
    needs_refinement: bool
    iteration_count: int
    max_iterations: int
