"""
State definition for the Critic Agent.
"""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class StepCritique(TypedDict):
    """Represents the critique of a single solution step."""
    step_number: int
    step_description: str
    step_content: str
    is_logically_correct: bool
    logic_feedback: str
    is_calculation_correct: bool
    calculation_feedback: str
    knowledge_points: List[str]


class CriticState(TypedDict):
    """State for the Critic Agent graph."""
    math_problem: str
    solution_steps: List[dict]  # List of steps up to and including the target step
    target_step_index: int  # Index of the step being critiqued (0-based)
    messages: Annotated[List[BaseMessage], add_messages]  # Message history for tool calling
    critique: Optional[StepCritique]
