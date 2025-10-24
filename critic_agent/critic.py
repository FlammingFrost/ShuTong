"""
Critic Agent implementation using LangGraph and GPT-4o.
"""

import os
import json
import re
from typing import List, Dict, Optional, Annotated, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from .state import CriticState, StepCritique
from .calculator import MathCalculator

# Load environment variables from .env file
load_dotenv()


class Critic:
    """
    Critic Agent that evaluates solution steps from the Solver Agent.
    
    Examines logical correctness, calculation accuracy, and identifies knowledge points
    for each step in a mathematical solution. Uses a calculator tool for verification.
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3, api_key: str = None, tracker=None):
        """
        Initialize the Critic Agent.
        
        Args:
            model_name: The OpenAI model to use (default: gpt-4o)
            temperature: Temperature for generation (default: 0.3, lower for more consistent evaluation)
            api_key: OpenAI API key (optional, will use environment variable if not provided)
            tracker: Tracker instance for logging tool usage (optional)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.tracker = tracker
        
        # Initialize the calculator for mathematical verification
        self.calculator = MathCalculator()
        
        # Create calculator tools
        self.tools = self._create_tools()
        
        # Initialize the language model with tools
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        if api_key:
            llm_kwargs["api_key"] = api_key
            
        self.llm = ChatOpenAI(**llm_kwargs)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
    
    def _create_tools(self) -> List:
        """
        Create calculator tools for the LLM to use.
        
        Returns:
            List of tool definitions
        """
        calculator = self.calculator
        
        @tool
        def evaluate_numerical(expression: str, variables: Optional[Dict[str, Any]] = None) -> str:
            """
            Evaluate a mathematical expression numerically.
            
            Args:
                expression: Mathematical expression as string (e.g., "2*x + 3")
                variables: Dictionary of variable values (e.g., {"x": 5} or {"x": 5.0})
                
            Returns:
                String with the numerical result or error message
                
            Examples:
                evaluate_numerical("2*5 + 3") -> "13.0"
                evaluate_numerical("2*x + 3", {"x": 5}) -> "13.0"
                evaluate_numerical("sin(pi/2)") -> "1.0"
            """
            try:
                # Convert to float if needed
                if variables:
                    float_variables = {k: float(v) if isinstance(v, (int, float, str)) else v 
                                      for k, v in variables.items()}
                else:
                    float_variables = None
                result = calculator.evaluate_numerical(expression, float_variables)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def evaluate_symbolic(expression: str, variables: Optional[Dict[str, Any]] = None, simplify: bool = False) -> str:
            """
            Evaluate a mathematical expression symbolically.
            
            Args:
                expression: Mathematical expression as string
                variables: Dictionary of variable values (can be symbolic strings or numeric values)
                simplify: Whether to simplify the result
                
            Returns:
                String with the symbolic result or error message
                
            Examples:
                evaluate_symbolic("x**2 + 2*x + 1") -> "x**2 + 2*x + 1"
                evaluate_symbolic("(x + 1)**2", simplify=True) -> "x**2 + 2*x + 1"
                evaluate_symbolic("2*x + 3", {"x": 5}) -> "13"
            """
            try:
                # Convert variables to strings if needed for symbolic evaluation
                if variables:
                    str_variables = {k: str(v) for k, v in variables.items()}
                else:
                    str_variables = None
                result = calculator.evaluate_symbolic(expression, str_variables, simplify)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def verify_calculation(expression: str, expected_result: str, variables: Optional[Dict[str, Any]] = None) -> str:
            """
            Verify if a calculation is correct by checking if expression equals expected result.
            
            Args:
                expression: Mathematical expression to evaluate
                expected_result: Expected result (can be numeric or symbolic string)
                variables: Optional variable values for substitution
                
            Returns:
                String indicating whether the calculation is correct with explanation
                
            Examples:
                verify_calculation("2 + 2", "4") -> "✓ Calculation correct: 2 + 2 = 4.0"
                verify_calculation("x**2 + 2*x + 1", "(x+1)**2") -> "✓ Expressions are mathematically equivalent"
                verify_calculation("2*x + 3", "13", {"x": 5}) -> "✓ Calculation correct: 2*x + 3 = 13.0"
            """
            try:
                # Try to convert expected_result to float if it's numeric
                try:
                    expected = float(expected_result)
                except ValueError:
                    expected = expected_result
                
                # Convert variables to float if provided
                if variables:
                    float_variables = {k: float(v) if isinstance(v, (int, float, str)) else v 
                                      for k, v in variables.items()}
                else:
                    float_variables = None
                
                is_correct, message = calculator.verify_calculation(expression, expected, float_variables)
                return message
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def compare_expressions(expr1: str, expr2: str, variables: Optional[Dict[str, Any]] = None) -> str:
            """
            Compare two mathematical expressions for equivalence.
            
            Args:
                expr1: First expression
                expr2: Second expression
                variables: Optional variable values for numeric comparison
                
            Returns:
                String indicating whether expressions are equivalent
                
            Examples:
                compare_expressions("x**2 + 2*x + 1", "(x+1)**2") -> "✓ Expressions are symbolically equivalent"
                compare_expressions("2*x", "x + x") -> "✓ Expressions are symbolically equivalent"
            """
            try:
                # Convert variables to float if provided
                if variables:
                    float_variables = {k: float(v) if isinstance(v, (int, float, str)) else v 
                                      for k, v in variables.items()}
                else:
                    float_variables = None
                
                are_equal, message = calculator.compare_expressions(expr1, expr2, float_variables)
                return message
            except Exception as e:
                return f"Error: {str(e)}"
        
        return [evaluate_numerical, evaluate_symbolic, verify_calculation, compare_expressions]
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for critiquing solution steps with tool calling.
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(CriticState)
        
        # Add nodes
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("tools", self._tools_node)
        
        # Set entry point
        workflow.set_entry_point("critique")
        
        # Add conditional edge: if tools are called, go to tools node; otherwise end
        workflow.add_conditional_edges(
            "critique",
            self._should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # After tools, go back to critique
        workflow.add_edge("tools", "critique")
        
        return workflow.compile()
    
    def _tools_node(self, state: CriticState) -> CriticState:
        """
        Custom tools node that tracks tool usage.
        
        Args:
            state: Current critic state
            
        Returns:
            Updated state with tool results
        """
        messages = state.get("messages", [])
        
        # Get the last AI message with tool calls
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                last_ai_message = msg
                break
        
        if not last_ai_message:
            return state
        
        # Execute tools and track each call
        tool_node = ToolNode(self.tools)
        result_state = tool_node.invoke(state)
        
        # Track tool usage if tracker is available
        if self.tracker:
            for tool_call in last_ai_message.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                # Find the corresponding tool result
                tool_result = None
                tool_error = None
                for msg in result_state.get("messages", []):
                    if isinstance(msg, ToolMessage) and msg.tool_call_id == tool_call.get("id"):
                        tool_result = msg.content
                        # Check if the result indicates an error
                        if isinstance(tool_result, str) and tool_result.startswith("Error:"):
                            tool_error = tool_result
                        break
                
                # Track the tool call
                self._track_tool_call(tool_name, tool_args, tool_result, tool_error)
        
        return result_state
    
    def _track_tool_call(self, tool_name: str, tool_args: Dict, result: Any, error: Optional[str]) -> None:
        """
        Track a tool call for logging purposes.
        
        Args:
            tool_name: Name of the tool called
            tool_args: Arguments passed to the tool
            result: Result from the tool
            error: Error message if tool failed
        """
        if not self.tracker:
            return
        
        record = {
            "run_id": self.tracker.current_run_id,
            "name": tool_name,
            "timestamp": datetime.now().isoformat(),
            "function": "tool_call",
            "values": {
                "tool_args": tool_args,
                "result": str(result)[:500] if result else None,  # Truncate long results
                "success": error is None
            },
            "status": "success" if error is None else "error",
            "error": error
        }
        
        self.tracker._save_record(tool_name, record)
    
    def _should_continue(self, state: CriticState) -> str:
        """
        Determine whether to continue with tool calls or end.
        
        Args:
            state: Current state
            
        Returns:
            "tools" if there are tool calls to execute, "end" otherwise
        """
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return "end"
    
    def _critique_node(self, state: CriticState) -> CriticState:
        """
        Node that generates the critique for a solution step.
        Can call calculator tools multiple times as needed.
        
        Args:
            state: Current critic state
            
        Returns:
            Updated state with messages or critique
        """
        messages = state.get("messages", [])
        
        # If we don't have messages yet, initialize the conversation
        if not messages:
            math_problem = state["math_problem"]
            solution_steps = state["solution_steps"]
            target_step_index = state["target_step_index"]
            
            # Build the initial prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(math_problem, solution_steps, target_step_index)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
        
        # Call LLM with tools
        response = self.llm_with_tools.invoke(messages)
        
        # Add response to messages
        updated_messages = messages + [response]
        
        # Check if we have a final response (no tool calls)
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            # Parse the final critique
            critique = self._parse_critique(state, updated_messages)
            
            return {
                "math_problem": state["math_problem"],
                "solution_steps": state["solution_steps"],
                "target_step_index": state["target_step_index"],
                "messages": updated_messages,
                "critique": critique,
            }
        
        # Otherwise, return state with updated messages for tool execution
        return {
            "math_problem": state["math_problem"],
            "solution_steps": state["solution_steps"],
            "target_step_index": state["target_step_index"],
            "messages": updated_messages,
            "critique": state.get("critique"),
        }
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the critic agent."""
        return """You are a mathematical critic agent. Your task is to analyze ONE step of a solution and provide detailed feedback.

For each step, you should:
1. Evaluate the logical correctness of the reasoning
2. Verify any calculations using the calculator tools provided
3. Identify relevant knowledge points (mathematical concepts, theorems, formulas)

You have access to these calculator tools:
- evaluate_numerical: Evaluate expressions numerically
- evaluate_symbolic: Evaluate expressions symbolically
- verify_calculation: Check if a calculation is correct
- compare_expressions: Check if two expressions are equivalent

Use these tools as needed to verify calculations. You can call them multiple times.

After verifying all necessary calculations, provide your final critique using the following format:

===LOGICAL_CORRECTNESS===
[TRUE or FALSE]

===LOGICAL_FEEDBACK===
[Your detailed explanation of logical correctness. You can use LaTeX math notation with $ or $$ delimiters.]

===CALCULATION_CORRECTNESS===
[TRUE or FALSE]

===CALCULATION_FEEDBACK===
[Your explanation including tool verification results. You can use LaTeX math notation with $ or $$ delimiters.]

===KNOWLEDGE_POINTS===
[Comma-separated list of knowledge point tags, e.g., induction, gamma_distribution, transformation_of_variables]

Focus ONLY on the current step. Consider previous steps as context but critique only the current one."""
    
    def _build_user_prompt(self, math_problem: str, solution_steps: List[dict], target_step_index: int) -> str:
        """Build the user prompt with problem and steps."""
        target_step = solution_steps[target_step_index]
        
        # Build context of previous steps
        previous_steps_text = ""
        if target_step_index > 0:
            previous_steps_text = "\n\n".join([
                f"### Step {i+1}: {step['description']}\n{step['content']}"
                for i, step in enumerate(solution_steps[:target_step_index])
            ])
        
        user_message = f"""Original Problem:
{math_problem}
"""
        
        if previous_steps_text:
            user_message += f"""
Previous Steps:
{previous_steps_text}
"""
        
        user_message += f"""
Step to Critique:
### Step {target_step_index + 1}: {target_step['description']}
{target_step['content']}

Please analyze this step, using calculator tools to verify any calculations, then provide your critique in the structured format specified."""
        
        return user_message
    
    def _parse_critique(self, state: CriticState, messages: List[BaseMessage]) -> StepCritique:
        """Parse the final critique from messages."""
        target_step_index = state["target_step_index"]
        target_step = state["solution_steps"][target_step_index]
        
        # Get the last AI message
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            return self._fallback_critique(target_step, target_step_index, "No AI response found")
        
        try:
            # Parse the structured text response
            content = last_ai_message.content
            critique_data = self._parse_structured_critique(content)
            
            critique: StepCritique = {
                "step_number": target_step_index + 1,
                "step_description": target_step['description'],
                "step_content": target_step['content'],
                "is_logically_correct": critique_data.get("logic_correct", True),
                "logic_feedback": critique_data.get("logic_feedback", ""),
                "is_calculation_correct": critique_data.get("calculation_correct", True),
                "calculation_feedback": critique_data.get("calculation_feedback", ""),
                "knowledge_points": critique_data.get("knowledge_points", []),
            }
            
        except Exception as e:
            critique = self._fallback_critique(target_step, target_step_index, str(e))
        
        return critique
    
    def _parse_structured_critique(self, content: str) -> Dict:
        """
        Parse structured text critique response.
        
        Args:
            content: The LLM response content with structured format
            
        Returns:
            Dictionary with parsed critique data
        """
        # Extract sections using regex patterns
        logic_correct = self._extract_boolean_field(content, "LOGICAL_CORRECTNESS")
        logic_feedback = self._extract_text_field(content, "LOGICAL_FEEDBACK")
        
        calculation_correct = self._extract_boolean_field(content, "CALCULATION_CORRECTNESS")
        calculation_feedback = self._extract_text_field(content, "CALCULATION_FEEDBACK")
        
        knowledge_points_str = self._extract_text_field(content, "KNOWLEDGE_POINTS")
        # Parse comma-separated knowledge points
        knowledge_points = [kp.strip() for kp in knowledge_points_str.split(',') if kp.strip()]
        
        return {
            "logic_correct": logic_correct,
            "logic_feedback": logic_feedback,
            "calculation_correct": calculation_correct,
            "calculation_feedback": calculation_feedback,
            "knowledge_points": knowledge_points,
        }
    
    def _extract_boolean_field(self, content: str, field_name: str) -> bool:
        """
        Extract a boolean field from structured critique.
        
        Args:
            content: The full critique content
            field_name: Name of the field to extract
            
        Returns:
            Boolean value (defaults to True if not found or unclear)
        """
        pattern = rf"===\s*{field_name}\s*===\s*\n\s*(\w+)"
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        
        if match:
            value = match.group(1).strip().upper()
            return value == "TRUE"
        
        return True  # Default to True if not found
    
    def _extract_text_field(self, content: str, field_name: str) -> str:
        """
        Extract a text field from structured critique.
        
        Args:
            content: The full critique content
            field_name: Name of the field to extract
            
        Returns:
            Extracted text content
        """
        # Pattern to match from ===FIELD_NAME=== to the next ===...=== or end of string
        pattern = rf"===\s*{field_name}\s*===\s*\n(.*?)(?=\n===|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return ""  # Return empty string if not found
    
    def _fallback_critique(self, target_step: dict, target_step_index: int, error: str) -> StepCritique:
        """Create a fallback critique when parsing fails."""
        return {
            "step_number": target_step_index + 1,
            "step_description": target_step['description'],
            "step_content": target_step['content'],
            "is_logically_correct": True,
            "logic_feedback": f"Unable to parse critique properly. Error: {error}",
            "is_calculation_correct": True,
            "calculation_feedback": "Unable to verify calculations due to parsing error.",
            "knowledge_points": [],
        }
    
    def critique_step(
        self,
        math_problem: str,
        solution_steps: List[dict],
        target_step_index: int
    ) -> StepCritique:
        """
        Critique a specific step in a solution.
        
        Args:
            math_problem: The original math problem
            solution_steps: List of solution steps (from solver agent)
            target_step_index: Index of the step to critique (0-based)
            
        Returns:
            Critique of the specified step
            
        Raises:
            ValueError: If target_step_index is out of range
        """
        if target_step_index < 0 or target_step_index >= len(solution_steps):
            raise ValueError(
                f"target_step_index {target_step_index} is out of range. "
                f"Valid range: 0 to {len(solution_steps) - 1}"
            )
        
        # Initialize state
        initial_state: CriticState = {
            "math_problem": math_problem,
            "solution_steps": solution_steps[:target_step_index + 1],  # Include steps up to target
            "target_step_index": target_step_index,
            "messages": [],
            "critique": None,
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return result["critique"]
    
    def critique_all_steps(
        self,
        math_problem: str,
        solution_steps: List[dict]
    ) -> List[StepCritique]:
        """
        Critique all steps in a solution sequentially.
        
        Args:
            math_problem: The original math problem
            solution_steps: List of solution steps (from solver agent)
            
        Returns:
            List of critiques, one for each step
        """
        critiques = []
        
        for i in range(len(solution_steps)):
            critique = self.critique_step(math_problem, solution_steps, i)
            critiques.append(critique)
        
        return critiques
    
    def format_critique(self, critique: StepCritique) -> str:
        """
        Format a critique as readable text.
        
        Args:
            critique: The critique to format
            
        Returns:
            Formatted critique string
        """
        lines = [
            f"{'='*80}",
            f"Critique of Step {critique['step_number']}: {critique['step_description']}",
            f"{'='*80}",
            "",
            "Logical Correctness:",
            f"  Status: {'✓ Correct' if critique['is_logically_correct'] else '✗ Incorrect'}",
            f"  Feedback: {critique['logic_feedback']}",
            "",
            "Calculation Correctness:",
            f"  Status: {'✓ Correct' if critique['is_calculation_correct'] else '✗ Incorrect'}",
            f"  Feedback: {critique['calculation_feedback']}",
            "",
            "Knowledge Points:",
        ]
        
        for kp in critique['knowledge_points']:
            lines.append(f"  - {kp}")
        
        return "\n".join(lines)
