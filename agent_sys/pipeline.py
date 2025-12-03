"""
Agent System Pipeline using LangGraph.

This module integrates Solver and Critic agents into a unified workflow.
"""

import uuid
import logging
from typing import List, Dict, Optional, Callable, Any
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
        tracker_dir: Optional[str] = "./data/tracker",
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
            tracker_dir: Directory for tracker data (None to disable tracking)
            api_key: OpenAI API key (optional)
        """
        # Initialize tracker only if tracker_dir is provided
        if tracker_dir is not None:
            self.tracker = Tracker(data_dir=tracker_dir)
        else:
            self.tracker = None
        
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
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Streaming callback (set during run_stream)
        self._stream_callback: Optional[Callable[[str, Any], None]] = None
        
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
        
        # Emit streaming event
        self._emit_event('stage', {'stage': 'solving', 'message': 'Generating solution...'})
        
        # Define the solve function
        def _solve(problem: str) -> tuple:
            solution = self.solver.solve(problem)
            steps = self.solver.get_solution_steps()
            return solution, steps
        
        # Apply tracking only if tracker is enabled
        if self.tracker:
            _solve = self.tracker.track(
                name="generate_initial_solution",
                value={
                    "math_problem": "args[0]",
                    "solution_length": "len(ret[0][0])",
                    "num_steps": "len(ret[0][1])",
                    "solution_steps_detail": "ret[0][1]"
                }
            )(_solve)
        
        initial_solution, solution_steps = _solve(math_problem)
        
        # Emit solution steps with step_number included in each step
        for i, step in enumerate(solution_steps):
            step_data = {
                'step_number': i + 1,
                'description': step.get('description', f'Step {i+1}'),
                'content': step.get('content', '')
            }
            self._emit_event('solution_step', step_data)
        
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
        
        # Emit streaming event
        iteration = state.get("iteration_count", 0) + 1
        self._emit_event('stage', {
            'stage': 'critiquing',
            'message': f'Analyzing solution (iteration {iteration})...',
            'iteration': iteration
        })
        
        # Track this operation with detailed critique information
        def _critique_all(problem: str, steps: List[Dict]) -> tuple:
            critiques, token_usage = self.critic.critique_all_steps(problem, steps)
            return critiques, token_usage
        
        # Apply tracking only if tracker is enabled
        if self.tracker:
            _critique_all = self.tracker.track(
                name="critique_all_steps",
                value={
                    "math_problem": "args[0]",
                    "num_steps": "len(args[1])",
                    "num_critiques": "len(ret[0])",
                    "issues_found": "sum(1 for c in ret[0] if not c.get('is_logically_correct', True) or not c.get('is_calculation_correct', True))",
                    "critiques_detail": "ret[0]",
                    "token_usage": "ret[1]"
                }
            )(_critique_all)
        
        all_critiques, token_usage = _critique_all(math_problem, solution_steps)
        
        # Emit critique events
        for critique in all_critiques:
            self._emit_event('critique', critique)
        
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
        
        # Track feedback collection if tracker is enabled
        if self.tracker:
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
        iteration_count = state["iteration_count"] + 1
        
        # Emit streaming event with identified errors/feedbacks
        self._emit_event('stage', {
            'stage': 'refining',
            'message': f'Refining solution (iteration {iteration_count})...',
            'iteration': iteration_count,
            'errors': feedbacks  # Include identified errors from critiques
        })
        
        # Define the refine function
        def _refine(fb: List[str], iteration: int) -> tuple:
            refined = self.solver.refine(fb)
            refined_steps = self.solver.get_solution_steps()
            return refined, refined_steps
        
        # Apply tracking only if tracker is enabled
        if self.tracker:
            _refine = self.tracker.track(
                name="refine_solution",
                value={
                    "num_feedbacks": "len(args[0])",
                    "feedbacks_detail": "args[0]",
                    "iteration": "args[1]",
                    "solution_length": "len(ret[0][0])",
                    "num_steps": "len(ret[0][1])",
                    "refined_steps_detail": "ret[0][1]"
                }
            )(_refine)
        
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
    
    def _emit_event(self, event_type: str, data: Any) -> None:
        """
        Emit a streaming event if callback is set.
        
        Args:
            event_type: Type of event ('stage', 'solution_step', 'critique', etc.)
            data: Event data
        """
        if self._stream_callback:
            try:
                self._stream_callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Error in stream callback: {e}")
    
    def run_stream(
        self, 
        math_problem: str, 
        max_iterations: Optional[int] = None,
        callback: Optional[Callable[[str, Any], None]] = None,
        run_id: Optional[str] = None
    ) -> Dict:
        """
        Run the agent pipeline with streaming callbacks.
        
        Args:
            math_problem: The math problem to solve
            max_iterations: Maximum refinement iterations (overrides default)
            callback: Callback function(event_type, data) for streaming updates
            run_id: Optional run identifier (will be generated if not provided)
            
        Returns:
            Same as run() - dictionary with final results
        """
        # Set the callback
        self._stream_callback = callback
        
        try:
            # Emit initialization event
            self._emit_event('stage', {'stage': 'initializing', 'message': 'Starting pipeline...'})
            
            # Run the pipeline normally
            result = self.run(math_problem, max_iterations, run_id)
            
            # Emit completion event
            self._emit_event('stage', {'stage': 'completed', 'message': 'Pipeline completed'})
            
            return result
        finally:
            # Clear callback
            self._stream_callback = None
    
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
        
        # Set run_id in tracker if tracker is enabled
        if self.tracker:
            self.tracker.set_run_id(run_id)
        
        # Define the internal run function
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
            # Apply tracker decorator only if tracker is enabled
            if self.tracker:
                tracked_func = self.tracker.track(
                    name="pipeline_run",
                    value={
                        "run_id": "args[0]",
                        "math_problem": "args[1]",
                        "max_iterations": "args[2]",
                        "final_iteration_count": "ret[0]['iteration_count']",
                        "num_knowledge_points": "len(ret[0]['knowledge_points'])"
                    }
                )(_run_pipeline)
                output = tracked_func(run_id, math_problem, max_iterations)
            else:
                output = _run_pipeline(run_id, math_problem, max_iterations)
            return output
        finally:
            # Clear run_id after pipeline completes (only if tracker exists)
            if self.tracker:
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
