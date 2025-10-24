# Critic Agent

A LangGraph-based agent that evaluates and critiques mathematical solution steps with integrated calculator tools for verification.

## State

### CriticState
```python
class CriticState(TypedDict):
    math_problem: str                    # Original math problem
    solution_steps: List[dict]           # Steps up to and including target
    target_step_index: int               # Index of step being critiqued (0-based)
    messages: List[BaseMessage]          # Message history for tool calling
    critique: Optional[StepCritique]     # Generated critique output
```

### StepCritique
```python
class StepCritique(TypedDict):
    step_number: int                     # Step number (1-based)
    step_description: str                # Description of the step
    step_content: str                    # Full content of the step
    is_logically_correct: bool           # Logical correctness status
    logic_feedback: str                  # Detailed logical evaluation
    is_calculation_correct: bool         # Calculation correctness status
    calculation_feedback: str            # Detailed calculation verification
    knowledge_points: List[str]          # Mathematical concepts used
```

### Solution Step Format
```python
step = {
    "description": str,  # Brief description of what the step does
    "content": str       # Full mathematical content of the step
}
```

## API

### Initialization
```python
Critic(model_name="gpt-4o", temperature=0.3, api_key=None)
```

### Methods

#### `critique_step(math_problem, solution_steps, target_step_index) -> StepCritique`
Critique a specific step in a solution.

**Parameters:**
- `math_problem` (str): Original math problem
- `solution_steps` (List[dict]): List of solution steps
- `target_step_index` (int): Index of step to critique (0-based)

**Returns:** `StepCritique` dict with evaluation results

**Raises:** `ValueError` if index out of range

---

#### `critique_all_steps(math_problem, solution_steps) -> List[StepCritique]`
Critique all steps in a solution sequentially.

**Parameters:**
- `math_problem` (str): Original math problem
- `solution_steps` (List[dict]): List of solution steps

**Returns:** List of `StepCritique` dicts, one per step

---

#### `format_critique(critique) -> str`
Format a critique as readable text.

**Parameters:**
- `critique` (StepCritique): Critique to format

**Returns:** Formatted string with correctness status and feedback

## Usage

### Basic Usage
```python
from critic_agent import Critic

# Initialize
critic = Critic(model_name="gpt-4o", temperature=0.3)

# Define problem and steps
problem = "Solve: $x^2 - 5x + 6 = 0$"
steps = [
    {
        "description": "Factor the quadratic",
        "content": "$x^2 - 5x + 6 = (x-2)(x-3) = 0$"
    },
    {
        "description": "Solve for x",
        "content": "Therefore $x = 2$ or $x = 3$"
    }
]

# Critique single step
critique = critic.critique_step(problem, steps, target_step_index=1)

# Access results
print(f"Logically correct: {critique['is_logically_correct']}")
print(f"Calculation correct: {critique['is_calculation_correct']}")
print(f"Knowledge: {critique['knowledge_points']}")
```

### Critique All Steps
```python
# Critique entire solution
all_critiques = critic.critique_all_steps(problem, steps)

# Display formatted output
for critique in all_critiques:
    print(critic.format_critique(critique))
```

### Integration with Solver Agent
```python
from solver_agent import Solver
from critic_agent import Critic

solver = Solver()
critic = Critic()

# Solve and critique
problem = "Integrate: $∫ x^2 dx$"
solution = solver.solve(problem)
steps = solver.get_solution_steps()
critiques = critic.critique_all_steps(problem, steps)

# Check for errors
errors = [c for c in critiques 
          if not (c['is_logically_correct'] and c['is_calculation_correct'])]

if errors:
    print(f"Found {len(errors)} errors:")
    for err in errors:
        print(f"  Step {err['step_number']}: {err['step_description']}")
```

## Calculator Tools

The critic uses integrated SymPy-based calculator tools to verify mathematical expressions:
- **evaluate_numerical**: Numerical evaluation (e.g., `2*5 + 3`, `sin(pi/2)`)
- **evaluate_symbolic**: Symbolic evaluation with simplification
- **verify_calculation**: Check if expression equals expected result
- **compare_expressions**: Check mathematical equivalence

The LLM autonomously calls these tools as needed during critique.