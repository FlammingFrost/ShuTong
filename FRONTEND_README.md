# Streamlit Frontend for ShuTong

A web-based interface for the ShuTong math problem solver agent system.

## Features

### Main Page - Agent QA
- **Input math problems** with LaTeX support (using `$...$` for inline math and `$$...$$` for display math)
- **Configure** max refinement iterations
- **Run** the agent pipeline to generate step-by-step solutions
- **View results** with:
  - Summary metrics (steps, iterations, knowledge points)
  - Final solution with LaTeX rendering
  - Step-by-step analysis with foldable critiques
  - Logic and calculation correctness indicators
  - Knowledge points summary
  - Remaining issues (if any)

### Records Page - View History
- **Browse** all previous runs from the database
- **Select** a specific run to view detailed information
- **Explore** tracked operations including:
  - Run summary (iterations, knowledge points, problem text)
  - All tracked function calls with timestamps
  - Detailed values for each operation
  - Reconstructed solution path showing initial solution and refinements

## Running the App

### Prerequisites
Make sure you have the environment set up:
```bash
conda activate shutong
```

### Start the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Solving a Math Problem
1. Navigate to "Main Page - Agent QA"
2. Enter your math problem in the text area (supports LaTeX)
3. Adjust the max refinement iterations slider if needed
4. Click "Run Agent Pipeline"
5. Wait for the pipeline to complete (may take a few minutes)
6. Review the results:
   - Expand each step to see critiques
   - Check knowledge points identified
   - Review any remaining issues

### Viewing History
1. Navigate to "Records Page - View History"
2. Select a run from the dropdown (shows timestamp and problem preview)
3. Explore:
   - Run summary metrics
   - Original problem
   - All tracked operations
   - Reconstructed solution path with initial and refined solutions

## LaTeX Math Support

The app supports LaTeX math expressions:
- Inline: `$x^2 + y^2 = z^2$`
- Display: `$$\int_0^\infty e^{-x} dx = 1$$`

Streamlit automatically renders these using MathJax.

## Architecture

The frontend uses:
- **Streamlit** for the web interface
- **AgentPipeline** from `agent_sys` for running the solver and critic agents
- **Tracker** for storing and retrieving run history from SQLite database

## Notes

- All runs are automatically tracked in `./data/tracker/record.db`
- Each run generates a unique `run_id` for easy reference
- The tracker stores detailed information about each step in the pipeline
- You can view the same data both in the frontend and by querying the database directly
