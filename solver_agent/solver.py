"""
Solver Agent implementation using LangGraph and GPT-4o.
"""

import os
import re
import logging
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from .state import SolverState, SolutionStep

# Load environment variables from .env file
load_dotenv()


class Solver:
    """
    Solver Agent that generates step-by-step solutions to math problems.
    
    Uses GPT-4o to generate formatted solutions and can refine them based on feedback.
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7, api_key: str = None):
        """
        Initialize the Solver Agent.
        
        Args:
            model_name: The OpenAI model to use (default: gpt-4o)
            temperature: Temperature for generation (default: 0.7)
            api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize the language model
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        if api_key:
            llm_kwargs["api_key"] = api_key
            
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
        
        # Store current state
        self._current_state: SolverState = None
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for solving math problems.
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(SolverState)
        
        # Add the solve node
        workflow.add_node("solve", self._solve_node)
        
        # Set entry point
        workflow.set_entry_point("solve")
        
        # Add edge to end
        workflow.add_edge("solve", END)
        
        return workflow.compile()
    
    def _solve_node(self, state: SolverState) -> SolverState:
        """
        Node that generates or refines the solution.
        
        Args:
            state: Current solver state
            
        Returns:
            Updated state with solution
        """
        math_problem = state["math_problem"]
        feedbacks = state.get("feedbacks")
        
        if feedbacks:
            # Refine existing solution
            solution = self._generate_refined_solution(
                math_problem, 
                state["current_solution"], 
                feedbacks
            )
        else:
            # Generate initial solution
            solution = self._generate_initial_solution(math_problem)
        
        # Parse the solution into steps
        steps = self._parse_solution_steps(solution)
        self.logger.info(f"  [Solver] Parsed {len(steps)} steps from solution")
        
        return {
            "math_problem": math_problem,
            "current_solution": solution,
            "solution_steps": steps,
            "feedbacks": feedbacks,
        }
    
    def _generate_initial_solution(self, math_problem: str) -> str:
        """
        Generate initial solution for a math problem.
        
        Args:
            math_problem: The math problem to solve
            
        Returns:
            Formatted solution as a string
        """
        self.logger.info(f"  [Solver] Generating initial solution...")
        self.logger.info(f"  [Solver] Problem: {math_problem[:80]}...")
        
        system_prompt = """You are an expert mathematics tutor. Your task is to solve math problems step-by-step with clear explanations.

Format your solution as follows:
- Use '### Step [N]: [Description]' to indicate each step (e.g., '### Step 1: Understand the problem')
- Use LaTeX format for all mathematical expressions
- Use $$ ... $$ for block equations (displayed on separate lines)
- Use $ ... $ for inline equations (within text)
- Provide clear, detailed explanations for each step
- Start with '### Step 1: Understand the problem', analyze the problem before proceeding
- Show all intermediate calculations, do not include thinking processes
- State the final answer clearly with '### Step [N]: Final answer'

Example:
Problem: Proof that for any continues random variable X, $F(X)$ follows uniform distribution.

Solution:

### Step 1: Understand the problem
We need to prove that if $X$ is a continuous random variable with cumulative distribution function (CDF) $F$, then the random variable $F(X)$ is uniformly distributed on the interval $[0,1]$.

### Step 1: Calculate the CDF of F(X)
Denote $\mu = F(X)$. The cumulative distribution function (CDF) of $\mu$ is given by:
$$P(\mu \leq t) = P(F(X) \leq t) = P(X \leq F^{-1}(t)) = F(F^{-1}(t)) = t$$

### Step 2: Compare with uniform distribution
The CDF of a uniform distribution $U(0,1)$ is also given by $$P(U \leq t) = t$$ for $t$ in $[0,1]$.

### Step 3: Final answer
Thus, since the CDFs match, we conclude that $F(X)$ follows a uniform distribution on $[0,1]$."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please solve the following math problem:\n\n{math_problem}")
        ]
        
        self.logger.info(f"  [Solver] Calling LLM to generate solution...")
        response = self.llm.invoke(messages)
        self.logger.info(f"  [Solver] Solution generated ({len(response.content)} chars)")
        return response.content
    
    def _generate_refined_solution(
        self, 
        math_problem: str, 
        previous_solution: str, 
        feedbacks: List[str]
    ) -> str:
        """
        Generate refined solution based on feedback.
        
        Args:
            math_problem: The original math problem
            previous_solution: The previous solution
            feedbacks: List of feedback comments
            
        Returns:
            Refined solution as a string
        """
        self.logger.info(f"  [Solver] Refining solution based on {len(feedbacks)} feedback(s)...")
        for i, fb in enumerate(feedbacks, 1):
            self.logger.info(f"  [Solver]   Feedback {i}: {fb[:80]}...")
        
        system_prompt = """You are an expert mathematics tutor. Your task is to refine math solutions based on feedback.

Format your solution as follows:
- Use '### Step [N]: [Description]' to indicate each step (e.g., '### Step 1: Understand the problem')
- Use LaTeX format for all mathematical expressions
- Use $$ ... $$ for block equations (displayed on separate lines)
- Use $ ... $ for inline equations (within text)
- Provide clear, detailed explanations for each step
- Start with '### Step 1: Understand the problem', analyze the problem before proceeding
- Show all intermediate calculations, do not include thinking processes
- State the final answer clearly with '### Step [N]: Final answer'"""

        feedback_text = "\n".join([f"- {fb}" for fb in feedbacks])
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Original Problem:
{math_problem}

Previous Solution:
{previous_solution}

Feedback to address:
{feedback_text}

Please provide a refined solution that addresses all the feedback points.""")
        ]
        
        self.logger.info(f"  [Solver] Calling LLM to refine solution...")
        response = self.llm.invoke(messages)
        self.logger.info(f"  [Solver] Refined solution generated ({len(response.content)} chars)")
        return response.content
    
    def _parse_solution_steps(self, solution: str) -> List[SolutionStep]:
        """
        Parse the solution text into structured steps.
        
        Args:
            solution: The formatted solution text
            
        Returns:
            List of solution steps with description and content
        """
        steps = []
        
        # Split by step headers (### Step [N]: [Description])
        step_pattern = r'###\s*Step\s*(\d+):\s*(.+?)(?=###\s*Step\s*\d+:|$)'
        matches = re.finditer(step_pattern, solution, re.DOTALL)
        
        for match in matches:
            step_num = match.group(1)
            rest = match.group(2).strip()
            
            # Split description from content (first line vs rest)
            lines = rest.split('\n', 1)
            description = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ""
            
            # If description contains the full content, split more intelligently
            if not content and '\n' in rest:
                parts = rest.split('\n', 1)
                description = parts[0].strip()
                content = parts[1].strip()
            elif not content:
                # Single line step
                content = description
                description = f"Step {step_num}"
            
            steps.append({
                "description": description,
                "content": content
            })
        
        # If no steps found, treat entire solution as one step
        if not steps:
            steps.append({
                "description": "Complete Solution",
                "content": solution.strip()
            })
        
        return steps
    
    def solve(self, math_problem: str) -> str:
        """
        Generate a step-by-step solution for a math problem.
        
        Args:
            math_problem: The math problem to solve in markdown format
            
        Returns:
            Step-by-step solution in markdown format with LaTeX equations
        """
        # Initialize state
        initial_state: SolverState = {
            "math_problem": math_problem,
            "current_solution": "",
            "solution_steps": [],
            "feedbacks": None,
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Store the current state
        self._current_state = result
        
        return result["current_solution"]
    
    def refine(self, feedbacks: List[str]) -> str:
        """
        Refine the current solution based on feedback.
        
        Args:
            feedbacks: List of feedback comments to address
            
        Returns:
            Refined solution in markdown format with LaTeX equations
            
        Raises:
            ValueError: If no solution has been generated yet
        """
        if self._current_state is None:
            raise ValueError("No solution to refine. Call solve() first.")
        
        # Update state with feedbacks
        refine_state: SolverState = {
            "math_problem": self._current_state["math_problem"],
            "current_solution": self._current_state["current_solution"],
            "solution_steps": self._current_state["solution_steps"],
            "feedbacks": feedbacks,
        }
        
        # Run the graph with feedback
        result = self.graph.invoke(refine_state)
        
        # Update current state
        self._current_state = result
        
        return result["current_solution"]
    
    def get_solution_steps(self) -> List[SolutionStep]:
        """
        Get the parsed solution steps from the current solution.
        
        Returns:
            List of solution steps with description and content
            
        Raises:
            ValueError: If no solution has been generated yet
        """
        if self._current_state is None:
            raise ValueError("No solution available. Call solve() first.")
        
        return self._current_state["solution_steps"]
    
    def get_current_state(self) -> SolverState:
        """
        Get the current solver state.
        
        Returns:
            Current state dictionary
            
        Raises:
            ValueError: If no solution has been generated yet
        """
        if self._current_state is None:
            raise ValueError("No state available. Call solve() first.")
        
        return self._current_state
