"""
Flask API Backend for ShuTong Frontend

This API provides endpoints for:
1. Getting analysis data for the overview charts
2. Generating math problems
3. Running the agent pipeline (with optional streaming)
4. Getting run history and details
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import sys
from pathlib import Path
import time
import logging
import queue
import threading
from typing import Generator

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent_sys import AgentPipeline
from tracker.tracker import Tracker
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import analysis utilities
from eval.ProcessBench.analyze import load_results_from_run, calculate_match_metrics

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize pipeline with tracker disabled
logger.info("Initializing AgentPipeline...")
pipeline = AgentPipeline(
    solver_model="gpt-5-mini",
    critic_model="gpt-5.1-2025-11-13",
    max_iterations=3,
    tracker_dir=None  # Disable tracker
)
logger.info("AgentPipeline initialized successfully")

# Keep tracker for history endpoints (but won't be used during pipeline runs)
tracker = Tracker(data_dir="./data/tracker")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'ShuTong API'})


@app.route('/api/analysis', methods=['GET'])
def get_analysis():
    """
    Get analysis data for overview charts.
    Reads from results directory and calculates metrics.
    """
    try:
        results_dir = project_root / 'results'
        
        # Get all model directories
        model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        
        models_data = []
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            
            try:
                # Load and calculate metrics for this model
                results = load_results_from_run(model_name, str(results_dir))
                metrics = calculate_match_metrics(results)
                
                models_data.append({
                    'name': model_name,
                    'displayName': get_model_display_name(model_name),
                    'exactAccuracy': metrics['exact_accuracy'],
                    'exactAccuracyPositive': metrics['exact_accuracy_positive'],
                    'exactAccuracyNegative': metrics['exact_accuracy_negative'],
                    'correctAccuracy': metrics['correct_accuracy'],
                    'correctPrecision': metrics['correct_precision'],
                    'correctRecall': metrics['correct_recall'],
                    'correctF1': metrics['correct_f1_score'],
                    'costPerExactMatch': calculate_cost_per_match(metrics, model_name, 'exact'),
                    'costPerCorrectMatch': calculate_cost_per_match(metrics, model_name, 'correct'),
                    'totalSamples': metrics['valid_samples'],
                })
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                continue
        
        return jsonify({'models': models_data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-problem', methods=['POST'])
def generate_problem():
    """Generate a math problem using LLM."""
    try:
        data = request.json
        topic = data.get('topic', '')
        difficulty = data.get('difficulty', 'Undergraduate')
        problem_type = data.get('problemType', 'Proof')
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-5.1-2025-11-13", temperature=0.8)
        
        # Create prompt
        system_prompt = """You are an expert mathematics professor. Your task is to generate ONE challenging and interesting math problem.

Generate a math problem following these requirements:
1. The problem should be short and well-defined. It should contained only 1 question and no sub-questions.
2. Use proper LaTeX notation for all mathematical expressions
3. Use $$ ... $$ for display equations (on separate lines)
4. Use $ ... $ for inline equations (within text)
5. Include any necessary context or definitions
6. Make the problem appropriately challenging for the specified level
7. Return ONLY the problem statement, no solutions or hints

Format the output as a clean markdown document with LaTeX math."""

        user_prompt = f"""Generate a {difficulty.lower()} level {problem_type.lower()} problem about: {topic}

Make it interesting, challenging, and mathematically rigorous."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Generate problem
        response = llm.invoke(messages)
        generated_problem = response.content
        
        return jsonify({
            'problem': generated_problem,
            'topic': topic,
            'difficulty': difficulty,
            'problemType': problem_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run-agent', methods=['POST'])
def run_agent():
    """Run the agent pipeline on a math problem."""
    try:
        data = request.json
        math_problem = data.get('math_problem', '')
        max_iterations = data.get('max_iterations', 3)
        
        if not math_problem:
            return jsonify({'error': 'Math problem is required'}), 400
        
        logger.info(f"Starting agent run with max_iterations={max_iterations}")
        logger.info(f"Problem: {math_problem[:100]}...")
        
        # Run the pipeline
        result = pipeline.run(
            math_problem=math_problem,
            max_iterations=max_iterations
        )
        
        logger.info(f"Agent completed with {result['iteration_count']} iterations")
        logger.info(f"Generated {len(result['solution_steps'])} solution steps")
        logger.info(f"Found {len(result['all_critiques'])} critiques")
        
        # Format solution steps with step numbers
        formatted_result = {
            **result,
            'solution_steps': [
                {
                    'step_number': i + 1,
                    'description': step.get('description', f'Step {i+1}'),
                    'content': step.get('content', '')
                }
                for i, step in enumerate(result['solution_steps'])
            ]
        }
        
        return jsonify(formatted_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run-agent-stream', methods=['POST'])
def run_agent_stream():
    """
    Run the agent pipeline with real-time streaming updates.
    Uses Server-Sent Events (SSE) to stream progress.
    """
    try:
        data = request.json
        math_problem = data.get('math_problem', '')
        max_iterations = data.get('max_iterations', 3)
        
        if not math_problem:
            return jsonify({'error': 'Math problem is required'}), 400
        
        def generate_events() -> Generator[str, None, None]:
            """Generator function for SSE with real-time updates."""
            # Create a thread-safe queue for events
            event_queue = queue.Queue()
            pipeline_result = {}
            pipeline_error = None
            
            def stream_callback(event_type: str, event_data: dict):
                """Handle streaming events from pipeline - called from pipeline thread."""
                event_queue.put((event_type, event_data))
            
            def run_pipeline_thread():
                """Run pipeline in background thread."""
                nonlocal pipeline_result, pipeline_error
                try:
                    logger.info("=" * 80)
                    logger.info(f"STREAMING: Starting agent pipeline")
                    logger.info(f"Problem: {math_problem[:100]}...")
                    logger.info(f"Max iterations: {max_iterations}")
                    logger.info("=" * 80)
                    
                    result = pipeline.run_stream(
                        math_problem=math_problem,
                        max_iterations=max_iterations,
                        callback=stream_callback
                    )
                    pipeline_result = result
                    logger.info(f"Pipeline completed! Iteration count: {result['iteration_count']}")
                    
                except Exception as e:
                    import traceback
                    pipeline_error = e
                    error_details = f"{str(e)}\n{traceback.format_exc()}"
                    logger.error("=" * 80)
                    logger.error(f"STREAMING ERROR: {error_details}")
                    logger.error("=" * 80)
                finally:
                    # Signal completion
                    event_queue.put(('__END__', None))
            
            # Start pipeline in background thread
            pipeline_thread = threading.Thread(target=run_pipeline_thread)
            pipeline_thread.daemon = True
            pipeline_thread.start()
            
            # Stream events as they arrive
            try:
                while True:
                    # Wait for next event (with timeout to keep connection alive)
                    try:
                        event_type, event_data = event_queue.get(timeout=1.0)
                    except queue.Empty:
                        # Send keepalive comment
                        yield ": keepalive\n\n"
                        continue
                    
                    # Check for completion signal
                    if event_type == '__END__':
                        break
                    
                    # Process and stream the event immediately
                    if event_type == 'stage':
                        stage = event_data.get('stage', 'unknown')
                        message = event_data.get('message', '')
                        logger.info(f"STREAM: Stage: {stage} - {message}")
                        
                        # Map internal stage names to frontend stages
                        if stage == 'solving':
                            yield f"data: {json.dumps({'stage': 'initial', 'message': message})}\n\n"
                        elif stage == 'critiquing':
                            yield f"data: {json.dumps({'stage': 'critique', 'message': message, 'iteration': event_data.get('iteration', 1)})}\n\n"
                        elif stage == 'refining':
                            refine_data = {
                                'stage': 'refine',
                                'message': message,
                                'iteration': event_data.get('iteration', 1)
                            }
                            # Include errors if present
                            if 'errors' in event_data:
                                refine_data['errors'] = event_data['errors']
                            yield f"data: {json.dumps(refine_data)}\n\n"
                        elif stage == 'initializing':
                            yield f"data: {json.dumps({'stage': 'initializing', 'message': message})}\n\n"
                    
                    elif event_type == 'solution_step':
                        step_num = event_data.get('step_number', 0)
                        logger.info(f"STREAM: Step {step_num}: {event_data.get('description', 'N/A')[:60]}...")
                        yield f"data: {json.dumps({'stage': 'solution_step', 'step': event_data})}\n\n"
                    
                    elif event_type == 'critique':
                        is_correct = event_data.get('is_logically_correct', True) and event_data.get('is_calculation_correct', True)
                        logger.info(f"STREAM: Step {event_data.get('step_number', '?')}: {'✓' if is_correct else '✗'}")
                        yield f"data: {json.dumps({'stage': 'critique_result', 'critique': event_data})}\n\n"
                
                # Wait for pipeline thread to complete
                pipeline_thread.join(timeout=5.0)
                
                # Send final result or error
                if pipeline_error:
                    yield f"data: {json.dumps({'stage': 'error', 'error': str(pipeline_error)})}\n\n"
                elif pipeline_result:
                    logger.info("STREAM: Sending final result...")
                    logger.info(f"Total knowledge points: {len(pipeline_result.get('knowledge_points', []))}")
                    formatted_result = {
                        **pipeline_result,
                        'solution_steps': [
                            {
                                'step_number': i + 1,
                                'description': step.get('description', f'Step {i+1}'),
                                'content': step.get('content', '')
                            }
                            for i, step in enumerate(pipeline_result['solution_steps'])
                        ]
                    }
                    yield f"data: {json.dumps({'stage': 'completed', 'result': formatted_result})}\n\n"
                    logger.info("=" * 80)
                    logger.info("STREAMING: Completed successfully")
                    logger.info("=" * 80)
                    
            except GeneratorExit:
                logger.warning("STREAM: Client disconnected")
            except Exception as e:
                import traceback
                logger.error(f"STREAM ERROR: {e}\n{traceback.format_exc()}")
                yield f"data: {json.dumps({'stage': 'error', 'error': str(e)})}\n\n"
        
        return Response(
            stream_with_context(generate_events()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo-result', methods=['GET'])
def get_demo_result():
    """Get pre-loaded demo result for quick demonstration."""
    try:
        # Demo data with complete solution
        demo_result = {
            'math_problem': r"""Let $f:\mathbb{R}\to\mathbb{R}$ be a twice differentiable function such that

- $f(0) = 0$ and $f(1) = 1$,
- for all $x \in \mathbb{R}$,
  $$
  0 \le f''(x) \le 2.
  $$

Compute the exact minimum possible value of the integral
$$
I = \int_0^1 \bigl(f(x)\bigr)^2\,dx
$$
over all such functions $f$, and justify that this minimum is attained.""",
            'final_solution': r"""The minimum possible value of $I = \int_0^1 (f(x))^2 \, dx$ under the given conditions is $\frac{1}{5}$, and this minimum is attained by the function $f(x) = x^2$.""",
            'iteration_count': 1,
            'solution_steps': [
                {
                    'step_number': 1,
                    'description': 'Understand the problem',
                    'content': r"""We are to minimize

$$
I = \int_0^1 (f(x))^2 \, dx
$$

over all twice differentiable functions $f:\mathbb{R}\to\mathbb{R}$ satisfying

$$
f(0) = 0, \quad f(1) = 1, \quad 0 \le f''(x) \le 2 \text{ for all } x \in \mathbb{R}.
$$

We must find the exact minimal value of $I$ and show that some admissible $f$ attains it."""
                },
                {
                    'step_number': 2,
                    'description': 'Compare $f$ with the parabola $x^2$',
                    'content': r"""Consider the function

$$
g(x) = f(x) - x^2.
$$

Compute its second derivative:

$$
g''(x) = f''(x) - 2.
$$

Since $0 \le f''(x) \le 2$, we have

$$
g''(x) = f''(x) - 2 \le 0 \text{ for all } x.
$$

Thus $g$ is concave on $[0,1]$. Also from the boundary conditions,

$$
g(0) = f(0) - 0 = 0, \quad g(1) = f(1) - 1 = 0.
$$"""
                },
                {
                    'step_number': 3,
                    'description': 'Use concavity to get a pointwise inequality',
                    'content': r"""For a concave function $g$ on $[0,1]$, for any $x \in [0,1]$ we have

$$
g(x) \ge (1-x)g(0) + x g(1) = 0.
$$

Hence

$$
g(x) \ge 0 \text{ for all } x \in [0,1],
$$

which means

$$
f(x) \ge x^2 \text{ for all } x \in [0,1].
$$"""
                },
                {
                    'step_number': 4,
                    'description': 'Lower bound for the integral',
                    'content': r"""Because $f(x) \ge x^2 \ge 0$ on $[0,1]$, squaring preserves the inequality:

$$
(f(x))^2 \ge x^4 \text{ for all } x \in [0,1].
$$

Integrating both sides gives

$$
I = \int_0^1 (f(x))^2 \, dx \ge \int_0^1 x^4 \, dx = \left[\frac{x^5}{5}\right]_0^1 = \frac{1}{5}.
$$"""
                },
                {
                    'step_number': 5,
                    'description': 'Attainment of the lower bound',
                    'content': r"""Equality in the chain above requires $g(x) \equiv 0$ on $[0,1]$, i.e. $f(x) = x^2$ for all $x \in [0,1]$. The function $f(x) = x^2$ satisfies the constraints:
- $f(0) = 0$, $f(1) = 1$,
- $f''(x) = 2$ for all $x$, so $0 \le f''(x) \le 2$ holds.

For this $f$,

$$
I = \int_0^1 (x^2)^2 \, dx = \int_0^1 x^4 \, dx = \frac{1}{5},
$$

so the lower bound is attained."""
                },
                {
                    'step_number': 6,
                    'description': 'Final answer',
                    'content': r"""The minimum possible value of

$$
I = \int_0^1 (f(x))^2 \, dx
$$

under the given conditions is

$$
\frac{1}{5},
$$

and this minimum is attained by the function $f(x) = x^2$."""
                }
            ],
            'all_critiques': [
                {
                    'step_number': 1,
                    'is_logically_correct': True,
                    'is_calculation_correct': True,
                    'logic_feedback': 'Correct',
                    'calculation_feedback': 'Correct',
                    'knowledge_points': ['calculus_of_variations', 'functional_optimization', 'integral_functional', 'constraint_optimization']
                },
                {
                    'step_number': 2,
                    'is_logically_correct': True,
                    'is_calculation_correct': True,
                    'logic_feedback': 'Correct',
                    'calculation_feedback': 'Correct',
                    'knowledge_points': ['convexity_concavity', 'second_derivative_test', 'boundary_conditions', 'parabola_comparison']
                },
                {
                    'step_number': 3,
                    'is_logically_correct': True,
                    'is_calculation_correct': True,
                    'logic_feedback': 'Correct',
                    'calculation_feedback': 'Correct',
                    'knowledge_points': ['concavity', 'convex_combinations', 'endpoint_values', 'pointwise_inequality', 'real_analysis_basics']
                },
                {
                    'step_number': 4,
                    'is_logically_correct': True,
                    'is_calculation_correct': True,
                    'logic_feedback': 'Correct',
                    'calculation_feedback': 'Correct',
                    'knowledge_points': ['integral_inequalities', 'order_preservation_by_squaring', 'definite_integrals', 'polynomial_integration', 'calculus_basics']
                },
                {
                    'step_number': 5,
                    'is_logically_correct': True,
                    'is_calculation_correct': True,
                    'logic_feedback': 'Correct',
                    'calculation_feedback': 'Correct',
                    'knowledge_points': ['convexity_and_concavity', 'inequalities', 'integral_calculus', 'boundary_conditions', 'optimization_over_functions']
                },
                {
                    'step_number': 6,
                    'is_logically_correct': True,
                    'is_calculation_correct': True,
                    'logic_feedback': 'Correct',
                    'calculation_feedback': 'Correct',
                    'knowledge_points': ['optimization_over_functions', 'convexity_and_concavity', 'integral_calculation', 'calculus_of_variations_basics']
                }
            ],
            'knowledge_points': [
                'calculus_of_variations',
                'functional_optimization',
                'integral_functional',
                'constraint_optimization',
                'convexity_concavity',
                'second_derivative_test',
                'boundary_conditions',
                'parabola_comparison',
                'concavity',
                'convex_combinations',
                'endpoint_values',
                'pointwise_inequality',
                'real_analysis_basics',
                'integral_inequalities',
                'order_preservation_by_squaring',
                'definite_integrals',
                'polynomial_integration',
                'calculus_basics',
                'inequalities',
                'integral_calculus',
                'optimization_over_functions',
                'integral_calculation',
                'calculus_of_variations_basics'
            ]
        }
        
        return jsonify(demo_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/runs', methods=['GET'])
def get_runs():
    """Get all run IDs from the tracker."""
    try:
        run_ids = tracker.get_all_run_ids()
        return jsonify({'runs': run_ids})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/runs/<run_id>', methods=['GET'])
def get_run_details(run_id):
    """Get detailed records for a specific run."""
    try:
        records = tracker.get_records(run_id=run_id)
        return jsonify({'run_id': run_id, 'records': records})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Helper functions

def get_model_display_name(model_key: str) -> str:
    """Get display name for a model."""
    model_names = {
        'gpt-4o-mini': 'GPT-4o Mini',
        'gpt-5-nano': 'GPT-5 Nano',
        'gpt-5-mini': 'GPT-5 Mini',
        'gpt-5.1': 'GPT-5.1',
        'gpt-5.1-2025-11-13': 'GPT-5.1(Reasoning)',
    }
    return model_names.get(model_key, model_key)


def calculate_cost_per_match(metrics: dict, model_name: str, match_type: str) -> float:
    """Calculate cost per match based on model pricing."""
    # Model pricing (per 1M tokens)
    pricing = {
        'gpt-4o-mini': {'input': 0.150, 'output': 0.600},
        'gpt-5-nano': {'input': 0.05, 'output': 0.4},
        'gpt-5-mini': {'input': 0.25, 'output': 2.00},
        'gpt-5.1': {'input': 1.25, 'output': 10.00},
        'gpt-5.1-2025-11-13': {'input': 1.25, 'output': 10.00},
    }
    
    # Default to GPT-4 pricing
    model_pricing = pricing.get(model_name, {'input': 2.50, 'output': 10.00})
    
    # Calculate total cost
    total_cost = (
        metrics['total_input_tokens'] / 1_000_000 * model_pricing['input'] +
        metrics['total_output_tokens'] / 1_000_000 * model_pricing['output']
    )
    
    # Calculate cost per match
    match_count = metrics['exact_match_count'] if match_type == 'exact' else metrics['correct_match_count']
    
    if match_count == 0:
        return 0.0
    
    return total_cost / match_count


if __name__ == '__main__':
    logger.info("="*80)
    logger.info("Starting ShuTong API server...")
    logger.info("API will be available at http://localhost:8000")
    logger.info("Tracker: DISABLED (for cleaner logging)")
    logger.info("="*80)
    app.run(debug=True, host='0.0.0.0', port=8000)
