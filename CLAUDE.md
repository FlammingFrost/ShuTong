# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ShuTong is an AI-powered math problem solver that uses a multi-agent system with LangGraph. The system consists of a **Solver Agent** that generates step-by-step solutions and a **Critic Agent** that evaluates and provides feedback. Solutions are iteratively refined based on critiques until no issues remain or max iterations are reached.

## Environment Setup

### Installation

Use Conda for environment management:

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate shutong
```

Alternatively, using pip:

```bash
# Install in Python 3.11+ environment
pip install -e .
```

### Configuration

- Create a `.env` file in the project root with your OpenAI API key:
  ```
  OPENAI_API_KEY=your_api_key_here
  ```

## Running the Application

### Streamlit Frontend

The main interface is a Streamlit app with three pages:
1. **Main Page - Agent QA**: Solve math problems with agent pipeline
2. **Problem Generator**: Generate math problems using AI
3. **Records Page**: View history of previous runs

```bash
# Run the Streamlit app
streamlit run app.py
```

The app will start on `http://localhost:8501` by default.

## Architecture

### Agent Pipeline Flow

The core workflow is implemented in `agent_sys/pipeline.py` using LangGraph:

1. **Generate Initial Solution** (`solver_agent/solver.py`)
   - Solver agent creates step-by-step solution with LaTeX formatting
   - Solution is parsed into structured steps (description + content)

2. **Critique All Steps** (`critic_agent/critic.py`)
   - Critic agent evaluates each step for logical and calculation correctness
   - Uses SymPy-based calculator tools for verification
   - Identifies knowledge points used in each step
   - Collects feedback for any issues found

3. **Refine Solution** (conditional, if issues found)
   - Solver receives all feedback and generates refined solution
   - Process repeats up to `max_iterations` times (default: 3)
   - Continues until no issues or iteration limit reached

4. **Return Final Result**
   - Final solution (refined or initial)
   - All critiques with detailed feedback
   - Collected knowledge points
   - Any remaining unresolved issues

### Key Components

#### AgentPipeline (`agent_sys/pipeline.py`)
- Main orchestrator using LangGraph StateGraph
- Coordinates Solver and Critic agents
- Manages iterative refinement loop
- Integrates with Tracker for comprehensive logging

#### Solver Agent (`solver_agent/solver.py`)
- Generates step-by-step solutions using GPT-4o
- Enforces structured format: `### Step [N]: [Description]`
- Supports refinement based on feedback
- Parses solutions into structured steps with regex

#### Critic Agent (`critic_agent/critic.py`)
- Evaluates individual steps or entire solutions
- Uses LangGraph with tool calling for calculator integration
- Returns `StepCritique` TypedDict with:
  - Logic correctness (bool + feedback)
  - Calculation correctness (bool + feedback)
  - Knowledge points (list of concepts)
- Calculator tools: `evaluate_numerical`, `evaluate_symbolic`, `verify_calculation`, `compare_expressions`

#### Tracker System (`tracker/tracker.py`)
- SQLite-based decorator system for recording function calls
- Supports custom value extraction with expressions
- Run-based grouping with `set_run_id()` / `clear_run_id()`
- Records include: timestamp, args/return values, status, errors
- Used throughout pipeline for comprehensive observability

### State Management

Each component uses TypedDict for state:
- **PipelineState** (`agent_sys/state.py`): Overall pipeline state
- **SolverState** (`solver_agent/state.py`): Solver agent state
- **CriticState** (`critic_agent/state.py`): Critic agent state

### Data Persistence

- **Tracker Database**: `./data/tracker/record.db` (SQLite)
  - Stores all function calls with detailed metrics
  - Indexed by `run_id` for efficient queries
  - Includes full solution steps and critique details

## Development Commands

### Testing

```bash
# Test individual components
python -m critic_agent.test_critic_with_tools
python -m critic_agent.test_calculator
python -m agent_sys.test_pipeline

# Run examples
python -m solver_agent.example
python -m agent_sys.example
python -m tracker.example
```

### Code Structure

```
ShuTong/
├── app.py                    # Streamlit frontend (main entry point)
├── agent_sys/                # Integrated pipeline
│   ├── pipeline.py          # AgentPipeline class (LangGraph workflow)
│   ├── state.py             # PipelineState TypedDict
│   └── example.py           # Usage examples
├── solver_agent/            # Solution generation
│   ├── solver.py            # Solver class
│   ├── state.py             # SolverState TypedDict
│   └── example.py
├── critic_agent/            # Solution evaluation
│   ├── critic.py            # Critic class (LangGraph with tools)
│   ├── calculator.py        # SymPy-based calculator tools
│   ├── state.py             # CriticState TypedDict
│   └── test_*.py            # Tests
└── tracker/                 # Function call tracking
    ├── tracker.py           # Tracker class with decorator
    └── example.py
```

## Important Implementation Details

### Solution Format Requirements

The Solver generates solutions using a specific format that must be maintained:

```markdown
### Step 1: Understand the problem
[Content explaining the problem]

### Step 2: [Description]
[Step content with LaTeX math using $ or $$]

### Step N: Final answer
[Final result]
```

This format is parsed using regex in `solver.py:_parse_solution_steps()`. Changes to the format require updating the parser.

### LaTeX Handling

- Solver outputs use `$` (inline) and `$$` (block) delimiters
- Streamlit frontend converts `\(` `\)` and `\[` `\]` to `$` `$$` in `app.py:convert_latex()`
- All mathematical expressions should use proper LaTeX syntax

### Tracker Value Extraction

The Tracker uses expression strings to extract values from function calls:
- `args[N]` - Access Nth argument
- `ret[0]` - Access return value (always wrapped in list)
- For tuple returns: `ret[0][0]`, `ret[0][1]`, etc.
- Supports dict/list operations: `args[0].get("key")`, `len(ret[0])`, etc.

Example:
```python
@tracker.track(
    name="generate_solution",
    value={
        "problem": "args[0]",
        "num_steps": "len(ret[0][1])",
        "steps_detail": "ret[0][1]"
    }
)
def solve(problem: str) -> tuple:
    solution = "..."
    steps = [...]
    return solution, steps
```

### Agent Configuration

Default models and temperatures:
- Solver: GPT-4o, temperature=0.7 (creative solutions)
- Critic: GPT-4o, temperature=0.3 (consistent evaluation)
- Max iterations: 3 (can be overridden per run)

### Critic Agent Design

The Critic operates in "stateless" mode - it evaluates each step independently based only on:
1. The original math problem
2. All steps up to and including the target step

This design choice means the Critic doesn't carry conversation history between steps, making evaluation more consistent but potentially less context-aware. When critiquing step N, it sees steps 1 through N.

## Data Directory

The `./data/tracker/` directory contains the SQLite database with all run history. This directory is gitignored. To reset history, delete `./data/tracker/record.db`.

## Branch Strategy

- **main**: Primary development branch
- **testing**: Current working branch (see git status)

When creating PRs, target the `main` branch.
