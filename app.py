"""
Streamlit Frontend for Agent System

This app provides a user interface for running the agent pipeline and viewing results.
"""

import streamlit as st
import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent_sys import AgentPipeline
from tracker.tracker import Tracker
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Helper function to convert LaTeX notation
def convert_latex(text):
    """Convert \\( \\) and \\[ \\] LaTeX delimiters to $ and $$ for Streamlit."""
    if not isinstance(text, str):
        return text
    # Convert \[ \] to $$
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)
    # Convert \( \) to $
    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)
    return text

# Page configuration
st.set_page_config(
    page_title="ShuTong - Math Agent System",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = AgentPipeline(
        solver_model="gpt-4o",
        critic_model="gpt-4o",
        max_iterations=3,
        tracker_dir="./data/tracker"
    )

if 'tracker' not in st.session_state:
    st.session_state.tracker = Tracker(data_dir="./data/tracker")

# Sidebar navigation
st.sidebar.title("🧮 ShuTong")
st.sidebar.markdown("Math Agent System")
page = st.sidebar.radio("Navigate", [
    "Main Page - Agent QA", 
    "Problem Generator", 
    "Records Page - View History"
])

# Main page content
if page == "Main Page - Agent QA":
    st.title("🎯 Math Problem Solver")
    st.markdown("Enter a math problem and let the agent system generate a step-by-step solution with critiques.")
    
    # Input area
    st.subheader("📝 Input Math Problem")
    
    # Check if there's a generated problem to use
    default_value = ""
    if 'generated_problem_for_qa' in st.session_state:
        default_value = st.session_state.generated_problem_for_qa
        st.info("📥 Problem loaded from Problem Generator! You can edit it below.")
        # Clear it after using once
        del st.session_state.generated_problem_for_qa
    
    math_problem = st.text_area(
        "Enter your math problem (supports LaTeX with $ or $$ delimiters):",
        value=default_value,
        height=200,
        placeholder="Example: Prove that for any continuous random variable X with CDF F(x), the random variable Y = F(X) follows a uniform distribution on [0,1]."
    )
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        max_iterations = st.slider("Max Refinement Iterations", 1, 5, 3)
    with col2:
        st.info(f"The solver will attempt up to {max_iterations} refinements based on critic feedback.")
    
    # Run button
    if st.button("🚀 Run Agent Pipeline", type="primary"):
        if not math_problem.strip():
            st.error("Please enter a math problem first!")
        else:
            with st.spinner("🔄 Running agent pipeline... This may take a few minutes."):
                try:
                    result = st.session_state.pipeline.run(
                        math_problem=math_problem,
                        max_iterations=max_iterations
                    )
                    
                    # Store result in session state
                    st.session_state.last_result = result
                    st.success(f"✅ Pipeline completed! ({result['iteration_count']} refinement iterations)")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        
        st.markdown("---")
        st.header("📊 Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Steps", len(result['solution_steps']))
        with col2:
            st.metric("Iterations", result['iteration_count'])
        with col3:
            st.metric("Knowledge Points", len(result['knowledge_points']))
        with col4:
            issues = len(result['feedbacks'])
            st.metric("Remaining Issues", issues, delta=None if issues == 0 else "⚠️")
        
        # Final solution
        st.subheader("📖 Final Solution")
        st.markdown(convert_latex(result['final_solution']))
        
        # Solution steps with critiques
        st.subheader("🔍 Step-by-Step Analysis")
        
        for i, (step, critique) in enumerate(zip(result['solution_steps'], result['all_critiques'])):
            with st.expander(f"**Step {i+1}: {convert_latex(step['description'])}**", expanded=False):
                # Step content
                st.markdown("**Content:**")
                st.markdown(convert_latex(step['content']))
                
                st.markdown("---")
                
                # Critique section
                st.markdown("**📝 Critique:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    logic_status = "✅ Correct" if critique['is_logically_correct'] else "❌ Incorrect"
                    st.markdown(f"**Logic:** {logic_status}")
                    st.markdown(convert_latex(critique['logic_feedback']))
                
                with col2:
                    calc_status = "✅ Correct" if critique['is_calculation_correct'] else "❌ Incorrect"
                    st.markdown(f"**Calculation:** {calc_status}")
                    st.markdown(convert_latex(critique['calculation_feedback']))
                
                # Knowledge points
                if critique['knowledge_points']:
                    st.markdown("**🏷️ Knowledge Points:**")
                    for kp in critique['knowledge_points']:
                        st.markdown(f"- {kp}")
        
        # Knowledge points summary
        st.subheader("📚 Knowledge Points Summary")
        if result['knowledge_points']:
            for i, kp in enumerate(result['knowledge_points'], 1):
                st.markdown(f"{i}. {kp}")
        else:
            st.info("No knowledge points identified.")
        
        # Remaining issues
        if result['feedbacks']:
            st.subheader("⚠️ Remaining Issues")
            st.warning(f"The following issues remain after {result['iteration_count']} iterations:")
            for fb in result['feedbacks']:
                st.markdown(f"- {fb}")

elif page == "Problem Generator":
    st.title("🎲 Math Problem Generator")
    st.markdown("Generate math problems on any topic using AI.")
    
    # Input area
    st.subheader("📚 Topic Input")
    topic = st.text_input(
        "Enter a math topic or concept:",
        placeholder="e.g., probability theory, calculus, linear algebra, differential equations"
    )
    
    # Additional options
    col1, col2 = st.columns(2)
    with col1:
        difficulty = st.selectbox(
            "Difficulty Level",
            ["Undergraduate", "Graduate", "Advanced Graduate", "Research Level"]
        )
    with col2:
        problem_type = st.selectbox(
            "Problem Type",
            ["Proof", "Calculation", "Application", "Conceptual", "Mixed"]
        )
    
    # Generate button
    if st.button("🎯 Generate Problem", type="primary"):
        if not topic.strip():
            st.error("Please enter a topic first!")
        else:
            with st.spinner("🤔 Generating math problem..."):
                try:
                    # Initialize LLM
                    llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
                    
                    # Create prompt
                    system_prompt = """You are an expert mathematics professor. Your task is to generate challenging and interesting math problems.

Generate a math problem following these requirements:
1. The problem should be clear and well-defined
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
                    
                    # Store in session state
                    st.session_state.generated_problem = generated_problem
                    st.success("✅ Problem generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating problem: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display generated problem
    if 'generated_problem' in st.session_state:
        st.markdown("---")
        st.subheader("📝 Generated Problem")
        
        # Editable text area
        edited_problem = st.text_area(
            "Edit the problem if needed (supports LaTeX):",
            value=st.session_state.generated_problem,
            height=300,
            key="problem_editor",
            help="You can edit the problem text here. Use $ for inline math and $$ for display math."
        )
        
        # Update session state if edited
        if edited_problem != st.session_state.generated_problem:
            st.session_state.generated_problem = edited_problem
        
        # Copy button
        st.code(edited_problem, language="markdown")
        
        # Rendered preview
        st.markdown("---")
        st.subheader("🎨 Rendered Preview")
        st.info("This is how the problem will appear with LaTeX rendering:")
        
        # Display with LaTeX conversion
        st.markdown(convert_latex(edited_problem))
        
        # Option to use in Agent QA
        st.markdown("---")
        if st.button("📤 Use This Problem in Agent QA"):
            st.session_state.generated_problem_for_qa = edited_problem
            st.success("✅ Problem saved! Switch to 'Main Page - Agent QA' to solve it.")
            st.info("💡 The problem will be pre-filled in the Main Page input area.")

elif page == "Records Page - View History":
    st.title("📜 Run History")
    st.markdown("View and explore previous runs of the agent system.")
    
    # Get all run IDs
    tracker = st.session_state.tracker
    run_ids = tracker.get_all_run_ids()
    
    if not run_ids:
        st.info("No runs found in the database yet. Run the agent pipeline on the main page to create records.")
    else:
        st.subheader(f"Found {len(run_ids)} runs")
        
        # Create a selection dropdown
        run_options = []
        run_map = {}
        
        for run_data in run_ids:
            run_id = run_data['run_id']
            timestamp = run_data['first_timestamp']
            problem = run_data.get('problem', '')
            
            # Create a display label
            problem_preview = problem[:80] + "..." if problem and len(problem) > 80 else problem or "No problem text"
            label = f"{timestamp} - {problem_preview}"
            
            run_options.append(label)
            run_map[label] = run_id
        
        selected_label = st.selectbox("Select a run to view details:", run_options)
        
        if selected_label:
            selected_run_id = run_map[selected_label]
            
            # Get all records for this run
            records = tracker.get_records(run_id=selected_run_id)
            
            st.markdown("---")
            st.header(f"🔍 Run Details: `{selected_run_id}`")
            
            # Find the pipeline_run record for summary
            pipeline_record = next((r for r in records if r['name'] == 'pipeline_run'), None)
            
            # 1. SUMMARY
            if pipeline_record:
                st.subheader("📊 Run Summary")
                values = pipeline_record.get('values', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Refinement Iterations", values.get('final_iteration_count', 'N/A'))
                with col2:
                    st.metric("Knowledge Points", values.get('num_knowledge_points', 'N/A'))
                with col3:
                    st.metric("Max Iterations", values.get('max_iterations', 'N/A'))
                
                # Display problem
                st.subheader("📝 Math Problem")
                problem_text = values.get('math_problem', 'No problem text found')
                st.markdown(convert_latex(problem_text))
            
            st.markdown("---")
            
            # 2. SOLUTION BY STEP WITH CRITIQUES
            st.subheader("📖 Solution Steps with Critiques")
            
            # Find critique records
            critique_records = [r for r in records if r['name'] == 'critique_all_steps']
            
            if critique_records:
                # Get the last (most recent) critique record which should have all critiques
                last_critique = critique_records[-1]
                critiques_detail = last_critique.get('values', {}).get('critiques_detail', [])
                
                if critiques_detail:
                    for critique in critiques_detail:
                        step_num = critique.get('step_number', 0)
                        step_desc = critique.get('step_description', 'Unknown step')
                        step_content = critique.get('step_content', '')
                        
                        # Display step content (with LaTeX support in title)
                        st.markdown(f"### Step {step_num}: {convert_latex(step_desc)}")
                        st.markdown(convert_latex(step_content))
                        
                        # Foldable critique section
                        with st.expander(f"📝 View Critique for Step {step_num}", expanded=False):
                            logic_correct = critique.get('is_logically_correct', True)
                            calc_correct = critique.get('is_calculation_correct', True)
                            
                            # Status indicators
                            col1, col2 = st.columns(2)
                            with col1:
                                logic_icon = "✅" if logic_correct else "❌"
                                st.markdown(f"**{logic_icon} Logic: {'Correct' if logic_correct else 'Incorrect'}**")
                                st.markdown(convert_latex(critique.get('logic_feedback', 'No feedback')))
                            
                            with col2:
                                calc_icon = "✅" if calc_correct else "❌"
                                st.markdown(f"**{calc_icon} Calculation: {'Correct' if calc_correct else 'Incorrect'}**")
                                st.markdown(convert_latex(critique.get('calculation_feedback', 'No feedback')))
                            
                            # Knowledge points
                            kps = critique.get('knowledge_points', [])
                            if kps:
                                st.markdown("**🏷️ Knowledge Points:**")
                                for kp in kps:
                                    st.markdown(f"- {kp}")
                        
                        st.markdown("---")
                else:
                    st.info("No detailed critiques found for this run.")
            else:
                st.info("No critique data available for this run.")
            
            # 3. ALL OTHER DETAILS
            st.subheader("🔧 Advanced Details")
            
            with st.expander("📋 View All Tracked Operations", expanded=False):
                st.markdown("Detailed logs of all operations performed during this run.")
                st.markdown("")
                
                # Group records by operation type
                operation_types = {}
                for record in records:
                    op_name = record['name']
                    if op_name not in operation_types:
                        operation_types[op_name] = []
                    operation_types[op_name].append(record)
            
                # Display each operation type
                for op_name, op_records in operation_types.items():
                    st.markdown(f"#### {op_name} ({len(op_records)} calls)")
                    
                    for i, record in enumerate(op_records):
                        st.markdown(f"**Call {i+1}** - {record['timestamp']} - Status: {record['status']}")
                        
                        if record.get('error'):
                            st.error(f"Error: {record['error']}")
                        
                        # Display tracked values (simplified view)
                        values = record.get('values', {})
                        if values:
                            for key, value in values.items():
                                if key.endswith('_detail'):
                                    # Skip detail fields in this simplified view
                                    continue
                                elif isinstance(value, (list, dict)):
                                    item_count = len(value) if isinstance(value, list) else "dict"
                                    st.markdown(f"- **{key}:** `[{item_count} items]`")
                                elif isinstance(value, str) and len(value) > 100:
                                    st.markdown(f"- **{key}:** `{value[:100]}...`")
                                else:
                                    st.markdown(f"- **{key}:** `{value}`")
                        
                        if i < len(op_records) - 1:
                            st.markdown("---")
                    
                    st.markdown("")
            
            with st.expander("🔄 View Solution Evolution", expanded=False):
                st.markdown("Track how the solution changed through refinement iterations.")
                st.markdown("")
                
                # Find initial and refined solutions
                initial_sol = next((r for r in records if r['name'] == 'generate_initial_solution'), None)
                refine_sols = [r for r in records if r['name'] == 'refine_solution']
                
                if initial_sol:
                    st.markdown("#### Initial Solution")
                    values = initial_sol.get('values', {})
                    st.markdown(f"**Number of steps:** {values.get('num_steps', 'N/A')}")
                    
                    if 'solution_steps_detail' in values:
                        steps = values['solution_steps_detail']
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"**Step {i}: {convert_latex(step.get('description', ''))}**")
                            st.markdown(convert_latex(step.get('content', '')))
                            st.markdown("")
                
                for i, refine_sol in enumerate(refine_sols, 1):
                    st.markdown(f"#### Refinement {i}")
                    values = refine_sol.get('values', {})
                    st.markdown(f"**Iteration:** {values.get('iteration', 'N/A')}")
                    st.markdown(f"**Number of feedbacks addressed:** {values.get('num_feedbacks', 'N/A')}")
                    
                    if 'feedbacks_detail' in values:
                        st.markdown("**Feedbacks that triggered this refinement:**")
                        for fb in values['feedbacks_detail']:
                            st.markdown(f"- {fb}")
                    
                    if 'refined_steps_detail' in values:
                        st.markdown("**Refined solution steps:**")
                        steps = values['refined_steps_detail']
                        for j, step in enumerate(steps, 1):
                            st.markdown(f"**Step {j}: {convert_latex(step.get('description', ''))}**")
                            st.markdown(convert_latex(step.get('content', '')))
                            st.markdown("")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**ShuTong** is an AI-powered math problem solver that uses "
    "a solver agent and critic agent to generate and refine solutions."
)
st.sidebar.markdown("Built with Streamlit, LangGraph, and GPT-4o")
