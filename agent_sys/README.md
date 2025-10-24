# Agent System

This folder builds the integrated agent system components using LangGraph.

## Components

### Solver Agent
- Provides step-by-step solutions to math problems
- Revises solutions based on feedback from critic agent
- Uses GPT-4o for generation

### Critic Agent
- Evaluates the quality of solutions provided by solver agents
- Provides constructive feedback for improvement
- Identifies knowledge points used in each step of the solution
- Uses calculator tools to verify calculations

### Knowledge Retriever
- **Not implemented yet**
- Will review the collection of knowledge points identified by critic agents
- Will combine similar points and output a unified knowledge base

## Pipeline Flow

The `AgentPipeline` class implements the following workflow using LangGraph:

```
[Math Problem] → Solver Agent → [Initial Solution]
                      ↓
        [Steps of Solution] (extracted automatically)
                      ↓
              For each Step:
        [Step] → Critic Agent → [Feedback] + [Knowledge Points]
                      ↓
    Collect all [Feedbacks] and [Knowledge Points]
                      ↓
          If issues found AND iterations remain:
    ([Initial Solution] + [Feedbacks]) → Solver Agent → [Refined Solution]
                      ↓
              Critique again (repeat until max iterations or no issues)
                      ↓
            [Final Solution] + [All Critiques] + [Knowledge Points]
```

## Usage

```python
from agent_sys import AgentPipeline

# Initialize pipeline (default max_iterations=3)
pipeline = AgentPipeline(
    solver_model="gpt-4o",
    critic_model="gpt-4o",
    max_iterations=3,  # Solver will attempt up to 3 refinements
    tracker_dir="./data/tracker"
)

# Run on a math problem
math_problem = "Your math problem here..."
result = pipeline.run(math_problem)

# Access results
print(pipeline.format_result(result))
final_solution = result["final_solution"]
knowledge_points = result["knowledge_points"]
critiques = result["all_critiques"]
```

## Features

- **Automatic Critique Loop**: Automatically critiques all solution steps
- **Iterative Refinement**: Refines solutions based on feedback (default: up to 3 iterations)
- **Knowledge Extraction**: Collects knowledge points from all solution steps
- **Comprehensive Tracking**: Records detailed information including:
  - Complete solution steps with descriptions and content
  - Detailed critiques for each step (logic, calculations, knowledge points)
  - Feedback content and refinement iterations
- **Flexible Configuration**: Customize models, temperatures, and iteration limits

## Files

- `pipeline.py`: Main `AgentPipeline` class implementing the LangGraph workflow
- `state.py`: `PipelineState` type definition for the pipeline
- `example.py`: Example usage patterns
- `README.md`: This file

## See Also

- `solver_agent/`: Solver agent implementation
- `critic_agent/`: Critic agent implementation
- `tracker/`: Tracker for recording function calls
