"""
State definition for the Solver Agent.
"""

from typing import TypedDict, List, Dict, Optional


class SolutionStep(TypedDict):
    """Represents a single step in the solution."""
    description: str
    content: str


class SolverState(TypedDict):
    """State for the Solver Agent graph."""
    math_problem: str
    current_solution: str
    solution_steps: List[SolutionStep]
    feedbacks: Optional[List[str]]
