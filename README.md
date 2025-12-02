# ShuTong

A multi-agent system for mathematical problem-solving that uses LangGraph to coordinate specialized AI agents. The system generates step-by-step solutions to math problems, evaluates them for correctness, and iteratively refines them based on constructive feedback.

## Overview

ShuTong employs three specialized agents working together:

- **Solver Agent**: Generates detailed, step-by-step solutions to mathematical problems using GPT-4o
- **Critic Agent**: Evaluates each solution step for logical and computational correctness, uses integrated calculator tools for verification, and identifies mathematical knowledge points
- **Knowledge Retriever** *(planned)*: Will aggregate and unify knowledge points identified across multiple problems

The system uses an iterative refinement loop where solutions are automatically improved based on critic feedback, with configurable iteration limits.

## Features

- **Automated Problem Solving**: Generate comprehensive step-by-step solutions to math problems
- **Intelligent Evaluation**: Each step is evaluated for both logical reasoning and computational accuracy
- **Tool-Augmented Verification**: Critic agent uses SymPy-based calculator tools to verify mathematical expressions
- **Iterative Refinement**: Solutions are automatically refined based on feedback (configurable iterations)
- **Knowledge Extraction**: Identifies and tracks mathematical concepts used in solutions
- **Interactive UI**: Streamlit-based web interface for easy interaction
- **Comprehensive Tracking**: SQLite-based tracking system records all agent interactions
- **Evaluation Framework**: Includes scripts for evaluating performance on ProcessBench dataset

## Prerequisites

- Python 3.11 or higher
- OpenAI API key (for GPT-4o access)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ShuTong

# Install dependencies with uv
uv sync

# Create .env file with your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Option 2: Using pip and venv

```bash
# Clone the repository
git clone <repository-url>
cd ShuTong

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install the package
pip install -e .

# Create .env file with your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Quick Start

### Using the Streamlit UI

The easiest way to use ShuTong is through the web interface:

```bash
streamlit run app.py
```

This will open a browser window where you can:
1. Enter a math problem
2. Configure solver and critic models
3. Set iteration limits
4. View the solution with step-by-step critiques
5. See extracted knowledge points

### Using the Python API

```python
from agent_sys import AgentPipeline

# Initialize the pipeline
pipeline = AgentPipeline(
    solver_model="gpt-4o",
    critic_model="gpt-4o",
    max_iterations=3,  # Number of refinement iterations
    tracker_dir="./data/tracker"
)

# Solve a problem
problem = "Solve the equation: $x^2 - 5x + 6 = 0$"
result = pipeline.run(problem)

# Access results
print(pipeline.format_result(result))
print(f"Final solution: {result['final_solution']}")
print(f"Knowledge points: {result['knowledge_points']}")
```

### Using Individual Agents

#### Solver Agent

```python
from solver_agent import Solver

solver = Solver(model_name="gpt-4o", temperature=0.7)
solution = solver.solve("Integrate: $\int x^2 dx$")
steps = solver.get_solution_steps()
```

#### Critic Agent

```python
from critic_agent import Critic

critic = Critic(model_name="gpt-4o", temperature=0.3)

# Critique a single step
critique = critic.critique_step(
    math_problem="Solve: $x^2 - 5x + 6 = 0$",
    solution_steps=steps,
    target_step_index=0
)

# Critique all steps
all_critiques = critic.critique_all_steps(
    math_problem="Solve: $x^2 - 5x + 6 = 0$",
    solution_steps=steps
)
```

## Running Tests

The project includes test scripts for verifying functionality:

```bash
# Test critic agent with calculator tools
python critic_agent/test_critic_with_tools.py

# Test calculator functionality
python critic_agent/test_calculator.py
python critic_agent/test_calculator_integration.py

# Test the full pipeline
python agent_sys/test_pipeline.py
```

## Evaluation

Evaluate the system on the ProcessBench dataset:

```bash
python eval/ProcessBench/eval_processbench.py
```

This script evaluates the critic agent's ability to identify errors in multi-step solutions.

## Project Structure

```
ShuTong/
├── agent_sys/          # Agent pipeline integration
│   ├── pipeline.py     # Main AgentPipeline class
│   ├── state.py        # Pipeline state definitions
│   └── example.py      # Usage examples
├── solver_agent/       # Solution generation agent
│   ├── solver.py       # Solver implementation
│   └── state.py        # Solver state definitions
├── critic_agent/       # Solution evaluation agent
│   ├── critic.py       # Critic implementation
│   ├── calculator.py   # Mathematical calculator tools
│   └── state.py        # Critic state definitions
├── tracker/            # Function call tracking system
│   └── tracker.py      # Tracker implementation
├── eval/               # Evaluation scripts
│   └── ProcessBench/   # ProcessBench dataset evaluation
├── data/               # Data storage (tracker DB)
├── app.py              # Streamlit web interface
└── pyproject.toml      # Project configuration
```

## Configuration

### Agent Configuration

Both agents can be configured with different models and parameters:

```python
pipeline = AgentPipeline(
    solver_model="gpt-4o",           # Or other OpenAI models
    critic_model="gpt-4o",
    solver_temperature=0.7,           # Higher = more creative
    critic_temperature=0.3,           # Lower = more consistent
    max_iterations=3,                 # Refinement iterations
    tracker_dir="./data/tracker"
)
```

### Critic Tool Iteration Limits

The critic agent limits tool-calling iterations to prevent infinite loops:

```python
critic = Critic(
    model_name="gpt-4o",
    temperature=0.3,
    max_tool_iterations=3  # Default: 3 rounds of tool calls per step
)
```

## How It Works

1. **Problem Input**: User provides a mathematical problem
2. **Initial Solution**: Solver agent generates a step-by-step solution
3. **Step Extraction**: Solution is parsed into individual steps
4. **Critique Phase**: Critic evaluates each step for:
   - Logical correctness
   - Computational accuracy (using calculator tools)
   - Knowledge points used
5. **Refinement Decision**: If errors found AND iterations remain:
   - Feedback is provided to solver
   - Solver generates refined solution
   - Process repeats
6. **Output**: Final solution with all critiques and knowledge points

## Calculator Tools

The critic agent has access to SymPy-based calculator tools that support:

- **Numerical evaluation**: `2*5 + 3`, `sin(pi/2)`
- **Symbolic evaluation**: Algebraic simplification
- **Expression verification**: Check if two expressions are equivalent
- **LaTeX support**: Parse and evaluate LaTeX mathematical expressions

## Data Storage

- **Tracker Database**: `./data/tracker/record.db` - SQLite database storing all function calls and tool usage
- **Evaluation Results**: `./results/` - Output from evaluation scripts

## Contributing

When contributing to this project:

1. Follow the existing code structure (LangGraph StateGraph pattern)
2. Define agent states as TypedDict in `state.py` files
3. Use the tracker system for logging important function calls
4. Add tests for new functionality
5. Update documentation as needed


## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration
- [SymPy](https://www.sympy.org/) - Symbolic mathematics
- [Streamlit](https://streamlit.io/) - Web interface
