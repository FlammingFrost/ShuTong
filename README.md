# ShuTong

An AI-powered mathematics problem solver using a multi-agent system with iterative refinement. ShuTong employs a Solver Agent to generate step-by-step solutions and a Critic Agent to evaluate and provide feedback, creating a loop of continuous improvement until solutions are verified or iteration limits are reached.

## Features

- **Multi-Agent Architecture**: Solver and Critic agents work together to produce high-quality solutions
- **Iterative Refinement**: Solutions are automatically refined based on critique feedback
- **Step-by-Step Solutions**: Clear, structured solutions with LaTeX-formatted mathematics
- **Calculation Verification**: Integrated SymPy-based calculator tools verify mathematical computations
- **Knowledge Extraction**: Automatically identifies and catalogs mathematical concepts used
- **Interactive Web Interface**: User-friendly Streamlit frontend with three main features:
  - Solve math problems with the agent pipeline
  - Generate practice problems using AI
  - View detailed history of previous runs
- **Comprehensive Tracking**: SQLite-based tracking system records all operations for analysis

## Prerequisites

- Python 3.11 or higher
- Conda (recommended) or pip
- OpenAI API key

## Installation

### Option 1: Using Conda (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd ShuTong
```

2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate shutong
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Option 2: Using pip

1. Clone the repository:
```bash
git clone <repository-url>
cd ShuTong
```

2. Create a virtual environment (Python 3.11+):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Running the Application

### Streamlit Web Interface

Launch the interactive web application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Using the API Programmatically

You can also use the agent system directly in Python:

```python
from agent_sys import AgentPipeline

# Initialize the pipeline
pipeline = AgentPipeline(
    solver_model="gpt-4o",
    critic_model="gpt-4o",
    max_iterations=3,
    tracker_dir="./data/tracker"
)

# Solve a math problem
problem = """
Prove that for any continuous random variable X with CDF F(x),
the random variable Y = F(X) follows a uniform distribution on [0,1].
"""

result = pipeline.run(problem)

# Access results
print(f"Iterations: {result['iteration_count']}")
print(f"Final Solution:\n{result['final_solution']}")
print(f"Knowledge Points: {result['knowledge_points']}")

# Format for display
print(pipeline.format_result(result))
```

## Project Structure

```
ShuTong/
├── app.py                      # Streamlit web interface
├── environment.yml             # Conda environment specification
├── pyproject.toml             # Project metadata and dependencies
├── README.md                  # This file
├── CLAUDE.md                  # Developer guidance for AI assistants
│
├── agent_sys/                 # Integrated agent pipeline
│   ├── __init__.py
│   ├── pipeline.py           # Main AgentPipeline orchestrator
│   ├── state.py              # Pipeline state definitions
│   ├── example.py            # Usage examples
│   └── test_pipeline.py      # Integration tests
│
├── solver_agent/             # Solution generation agent
│   ├── __init__.py
│   ├── solver.py             # Solver agent implementation
│   ├── state.py              # Solver state definitions
│   └── example.py            # Usage examples
│
├── critic_agent/             # Solution evaluation agent
│   ├── __init__.py
│   ├── critic.py             # Critic agent implementation
│   ├── calculator.py         # SymPy-based calculator tools
│   ├── state.py              # Critic state definitions
│   ├── test_critic_with_tools.py
│   └── test_calculator.py
│
└── tracker/                  # Function call tracking system
    ├── __init__.py
    ├── tracker.py            # Tracker implementation
    └── example.py            # Usage examples
```

## How It Works

### Agent Pipeline Flow

1. **Initial Solution Generation**
   - Solver Agent receives the math problem
   - Generates structured step-by-step solution with LaTeX formatting
   - Solution is parsed into individual steps

2. **Critique Phase**
   - Critic Agent evaluates each step independently
   - Checks logical correctness and calculation accuracy
   - Uses calculator tools to verify mathematical expressions
   - Identifies knowledge points (mathematical concepts) used
   - Generates specific feedback for any issues found

3. **Refinement Loop** (if issues detected)
   - Solver Agent receives all feedback
   - Generates refined solution addressing the issues
   - Returns to Critique Phase
   - Continues until no issues remain or max iterations reached

4. **Final Output**
   - Refined solution with all steps
   - Complete critiques for each step
   - Extracted knowledge points
   - Any remaining unresolved issues (if max iterations reached)

### Key Technologies

- **LangGraph**: Orchestrates multi-agent workflows with state management
- **LangChain**: Provides LLM integrations and tool calling capabilities
- **OpenAI GPT-4o**: Powers both Solver and Critic agents
- **SymPy**: Symbolic mathematics library for calculation verification
- **Streamlit**: Interactive web interface
- **SQLite**: Persistent storage for run tracking and history

## Configuration

### Agent Settings

You can customize agent behavior when initializing the pipeline:

```python
pipeline = AgentPipeline(
    solver_model="gpt-4o",           # Model for solution generation
    critic_model="gpt-4o",           # Model for evaluation
    solver_temperature=0.7,          # Higher = more creative solutions
    critic_temperature=0.3,          # Lower = more consistent evaluation
    max_iterations=3,                # Maximum refinement iterations
    tracker_dir="./data/tracker"     # Directory for tracking database
)
```

### Environment Variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Testing

Run tests for individual components:

```bash
# Test critic agent with calculator tools
python -m critic_agent.test_critic_with_tools

# Test calculator tools
python -m critic_agent.test_calculator

# Test full pipeline
python -m agent_sys.test_pipeline
```

Run examples:

```bash
# Solver agent example
python -m solver_agent.example

# Agent pipeline example
python -m agent_sys.example

# Tracker example
python -m tracker.example
```

## Data Persistence

The application stores execution history in `./data/tracker/record.db` (SQLite database). This includes:

- All function calls with timestamps
- Input parameters and return values
- Solution steps and critiques
- Performance metrics

To reset history, delete the database file:

```bash
rm ./data/tracker/record.db
```

## Troubleshooting

### API Key Issues

If you see authentication errors:
1. Verify your `.env` file exists in the project root
2. Check that `OPENAI_API_KEY` is set correctly
3. Ensure the API key has sufficient credits

### Import Errors

If you encounter import errors:
1. Verify the environment is activated: `conda activate shutong`
2. Reinstall in editable mode: `pip install -e .`
3. Check Python version: `python --version` (should be 3.11+)

### Streamlit Issues

If Streamlit won't start:
1. Check if port 8501 is available
2. Try specifying a different port: `streamlit run app.py --server.port 8502`
3. Clear Streamlit cache: `rm -rf ~/.streamlit/cache`

## Contributing

When contributing to this project:

1. Create a new branch from `main`
2. Make your changes
3. Test thoroughly using the test commands above
4. Create a pull request targeting `main`

## License

[Add license information]

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) for LLM integration
- [OpenAI](https://openai.com/) for GPT-4o model
- [Streamlit](https://streamlit.io/) for web interface
- [SymPy](https://www.sympy.org/) for symbolic mathematics
