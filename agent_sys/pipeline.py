"""
Agent System Pipeline using LangGraph.

This module integrates Solver and Critic agents into a unified workflow.
"""

import uuid
from typing import List, Dict, Optional
from langgraph.graph import StateGraph, END
from solver_agent.solver import Solver
from critic_agent.critic import Critic
from tracker.tracker import Tracker
from .state import PipelineState


class AgentPipeline:
    """
    Integrated pipeline that coordinates Solver and Critic agents.
    
    Pipeline flow:
    1. Solver generates initial solution
    2. Extract steps from the solution
    3. For each step:
       - Critic evaluates the step
       - Collect feedback and knowledge points
    4. If issues found, Solver refines solution based on feedback
    5. Return final solution with all critiques and knowledge points
    """
    
    def __init__(
        self,
        solver_model: str = "gpt-4o",
        critic_model: str = "gpt-4o",
        solver_temperature: float = 0.7,
        critic_temperature: float = 0.3,
        max_iterations: int = 1,
        tracker_dir: str = "./data/tracker",
        api_key: Optional[str] = None
    ):
        """
        Initialize the Agent Pipeline.
        
        Args:
            solver_model: Model name for solver agent
            critic_model: Model name for critic agent
            solver_temperature: Temperature for solver
            critic_temperature: Temperature for critic
            max_iterations: Maximum refinement iterations (default: 3)
            tracker_dir: Directory for tracker data
            api_key: OpenAI API key (optional)
        """
        # Initialize tracker first
        self.tracker = Tracker(data_dir=tracker_dir)
        
        # Initialize agents
        self.solver: Solver = Solver(
            model_name=solver_model,
            temperature=solver_temperature,
            api_key=api_key
        )

        self.critic: Critic = Critic(
            model_name=critic_model,
            temperature=critic_temperature,
            api_key=api_key,
            tracker=self.tracker
        )
        
        # Store configuration
        self.max_iterations = max_iterations
        
        # Build the pipeline graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for the agent pipeline.
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("generate_initial_solution", self._generate_initial_solution_node)
        workflow.add_node("critique_all_steps", self._critique_all_steps_node)
        workflow.add_node("refine_solution", self._refine_solution_node)
        
        # Set entry point
        workflow.set_entry_point("generate_initial_solution")
        
        # Add edges
        workflow.add_edge("generate_initial_solution", "critique_all_steps")
        
        # Conditional edge: refine if needed and iterations remain
        workflow.add_conditional_edges(
            "critique_all_steps",
            self._should_refine,
            {
                "refine": "refine_solution",
                "end": END
            }
        )
        
        # After refinement, critique again
        workflow.add_edge("refine_solution", "critique_all_steps")
        
        return workflow.compile()
    
    def _generate_initial_solution_node(self, state: PipelineState) -> Dict:
        """
        Node that generates the initial solution using Solver agent.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with initial solution
        """
        math_problem = state["math_problem"]
        
        # Track this operation with detailed step information
        @self.tracker.track(
            name="generate_initial_solution",
            value={
                "math_problem": "args[0]",
                "solution_length": "len(ret[0][0])",
                "num_steps": "len(ret[0][1])",
                "solution_steps_detail": "ret[0][1]"  # Track detailed steps
            }
        )
        def _solve(problem: str) -> tuple:
            solution = self.solver.solve(problem)
            steps = self.solver.get_solution_steps()
            return solution, steps
        
        initial_solution, solution_steps = _solve(math_problem)
        
        return {
            "math_problem": math_problem,
            "initial_solution": initial_solution,
            "solution_steps": solution_steps,
            "all_critiques": [],
            "feedbacks": [],
            "knowledge_points": [],
            "refined_solution": None,
            "refined_steps": None,
            "needs_refinement": False,
            "iteration_count": 0,
            "max_iterations": state.get("max_iterations", self.max_iterations)
        }
    
    def _critique_all_steps_node(self, state: PipelineState) -> Dict:
        """
        Node that critiques all solution steps using Critic agent.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with critiques, feedbacks, and knowledge points
        """
        math_problem = state["math_problem"]
        
        # Use refined solution if available, otherwise use initial
        if state.get("refined_solution"):
            solution_steps = state["refined_steps"]
        else:
            solution_steps = state["solution_steps"]
        
        # Track this operation with detailed critique information
        @self.tracker.track(
            name="critique_all_steps",
            value={
                "math_problem": "args[0]",
                "num_steps": "len(args[1])",
                "num_critiques": "len(ret[0])",
                "issues_found": "sum(1 for c in ret[0] if not c['is_logically_correct'] or not c['is_calculation_correct'])",
                "critiques_detail": "ret[0]"  # Track detailed critiques for each step
            }
        )
        def _critique_all(problem: str, steps: List[Dict]) -> List:
            critiques = self.critic.critique_all_steps(problem, steps)
            return critiques
        
        all_critiques = _critique_all(math_problem, solution_steps)
        
        # Extract feedbacks and knowledge points
        feedbacks = []
        knowledge_points = []
        needs_refinement = False
        
        for critique in all_critiques:
            # Collect feedback for issues
            if not critique["is_logically_correct"]:
                feedbacks.append(
                    f"Step {critique['step_number']} - Logic Issue: {critique['logic_feedback']}"
                )
                needs_refinement = True
            
            if not critique["is_calculation_correct"]:
                feedbacks.append(
                    f"Step {critique['step_number']} - Calculation Issue: {critique['calculation_feedback']}"
                )
                needs_refinement = True
            
            # Collect knowledge points
            for kp in critique["knowledge_points"]:
                if kp not in knowledge_points:
                    knowledge_points.append(kp)
        
        # Track feedback collection
        @self.tracker.track(
            name="collect_feedback_and_knowledge",
            value={
                "num_feedbacks": "len(args[0])",
                "num_knowledge_points": "len(args[1])",
                "needs_refinement": "args[2]"
            }
        )
        def _collect(fb: List[str], kp: List[str], needs_ref: bool) -> tuple:
            return fb, kp, needs_ref
        
        _collect(feedbacks, knowledge_points, needs_refinement)
        
        return {
            "all_critiques": all_critiques,
            "feedbacks": feedbacks,
            "knowledge_points": knowledge_points,
            "needs_refinement": needs_refinement
        }
    
    def _refine_solution_node(self, state: PipelineState) -> Dict:
        """
        Node that refines the solution based on feedback.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with refined solution
        """
        feedbacks = state["feedbacks"]
        
        # Track this operation with detailed step information
        @self.tracker.track(
            name="refine_solution",
            value={
                "num_feedbacks": "len(args[0])",
                "feedbacks_detail": "args[0]",  # Track actual feedback content
                "iteration": "args[1]",
                "solution_length": "len(ret[0][0])",
                "num_steps": "len(ret[0][1])",
                "refined_steps_detail": "ret[0][1]"  # Track detailed refined steps
            }
        )
        def _refine(fb: List[str], iteration: int) -> tuple:
            refined = self.solver.refine(fb)
            refined_steps = self.solver.get_solution_steps()
            return refined, refined_steps
        
        iteration_count = state["iteration_count"] + 1
        refined_solution, refined_steps = _refine(feedbacks, iteration_count)
        
        return {
            "refined_solution": refined_solution,
            "refined_steps": refined_steps,
            "iteration_count": iteration_count,
            "needs_refinement": False,  # Reset flag
            "feedbacks": [],  # Clear feedbacks for next iteration
        }
    
    def _should_refine(self, state: PipelineState) -> str:
        """
        Determine whether to refine the solution or end.
        
        Args:
            state: Current pipeline state
            
        Returns:
            "refine" if refinement is needed and iterations remain, "end" otherwise
        """
        needs_refinement = state.get("needs_refinement", False)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)
        
        if needs_refinement and iteration_count < max_iterations:
            return "refine"
        return "end"
    
    def run(self, math_problem: str, max_iterations: Optional[int] = None, run_id: Optional[str] = None) -> Dict:
        """
        Run the agent pipeline on a math problem.
        
        Args:
            math_problem: The math problem to solve
            max_iterations: Maximum refinement iterations (overrides default)
            run_id: Optional run identifier (will be generated if not provided)
            
        Returns:
            Dictionary containing:
                - run_id: The run identifier for this execution
                - initial_solution: Initial solution from solver
                - final_solution: Final solution (refined or initial)
                - solution_steps: Final solution steps
                - all_critiques: List of critiques for final solution
                - feedbacks: List of remaining feedbacks (if max iterations reached)
                - knowledge_points: Collected knowledge points
                - iteration_count: Number of refinement iterations performed
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        # Set run_id in tracker
        self.tracker.set_run_id(run_id)
        
        # Track the full pipeline run
        @self.tracker.track(
            name="pipeline_run",
            value={
                "run_id": "args[0]",
                "math_problem": "args[1]",
                "max_iterations": "args[2]",
                "final_iteration_count": "ret[0]['iteration_count']",
                "num_knowledge_points": "len(ret[0]['knowledge_points'])"
            }
        )
        def _run_pipeline(rid: str, problem: str, max_iter: Optional[int]) -> Dict:
            # Initialize state
            initial_state: PipelineState = {
                "math_problem": problem,
                "initial_solution": "",
                "solution_steps": [],
                "all_critiques": [],
                "feedbacks": [],
                "knowledge_points": [],
                "refined_solution": None,
                "refined_steps": None,
                "needs_refinement": False,
                "iteration_count": 0,
                "max_iterations": max_iter if max_iter is not None else self.max_iterations
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Prepare output
            output = {
                "run_id": rid,
                "initial_solution": result["initial_solution"],
                "final_solution": result.get("refined_solution") or result["initial_solution"],
                "solution_steps": result.get("refined_steps") or result["solution_steps"],
                "all_critiques": result["all_critiques"],
                "feedbacks": result["feedbacks"],
                "knowledge_points": result["knowledge_points"],
                "iteration_count": result["iteration_count"]
            }
            
            return output
        
        try:
            output = _run_pipeline(run_id, math_problem, max_iterations)
            return output
        finally:
            # Clear run_id after pipeline completes
            self.tracker.clear_run_id()
    
    def get_solution_text(self, result: Dict) -> str:
        """
        Extract the final solution text from pipeline result.
        
        Args:
            result: Result dictionary from run()
            
        Returns:
            Final solution text
        """
        return result["final_solution"]
    
    def get_knowledge_points(self, result: Dict) -> List[str]:
        """
        Extract knowledge points from pipeline result.
        
        Args:
            result: Result dictionary from run()
            
        Returns:
            List of unique knowledge points
        """
        return result["knowledge_points"]
    
    def format_result(self, result: Dict) -> str:
        """
        Format the pipeline result as readable text.
        
        Args:
            result: Result dictionary from run()
            
        Returns:
            Formatted result string
        """
        lines = [
            "=" * 80,
            "AGENT PIPELINE RESULT",
            "=" * 80,
            "",
            f"Refinement Iterations: {result['iteration_count']}",
            "",
            "=" * 80,
            "FINAL SOLUTION",
            "=" * 80,
            "",
            result["final_solution"],
            "",
            "=" * 80,
            "KNOWLEDGE POINTS IDENTIFIED",
            "=" * 80,
            ""
        ]
        
        for i, kp in enumerate(result["knowledge_points"], 1):
            lines.append(f"{i}. {kp}")
        
        lines.extend([
            "",
            "=" * 80,
            "CRITIQUES OF FINAL SOLUTION",
            "=" * 80,
            ""
        ])
        
        for critique in result["all_critiques"]:
            lines.append(self.critic.format_critique(critique))
            lines.append("")
        
        if result["feedbacks"]:
            lines.extend([
                "=" * 80,
                "REMAINING ISSUES (Max iterations reached)",
                "=" * 80,
                ""
            ])
            for fb in result["feedbacks"]:
                lines.append(f"- {fb}")
        
        return "\n".join(lines)
