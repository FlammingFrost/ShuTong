"""
Test the critic agent with tool calling using a probability problem.

Problem: Calculate joint cumulative distribution from joint density.
"""

from dotenv import load_dotenv
from critic_agent import Critic

load_dotenv()


def test_correct_solution():
    """Test critic agent with a correct solution."""
    
    print("=" * 80)
    print("TEST 1: Correct Solution")
    print("=" * 80)
    
    problem = """
Given the joint probability density function:
f(x,y) = 2 for 0 ≤ x ≤ y ≤ 1

Find the joint cumulative distribution function F(x,y) at the point (0.5, 0.5).
"""
    
    correct_steps = [
        {
            "description": "Set up the double integral for F(0.5, 0.5)",
            "content": """
The joint cumulative distribution function is defined as:
F(x,y) = ∫∫ f(s,t) ds dt, where the integration is over the region where s ≤ x and t ≤ y.

For F(0.5, 0.5), we need:
F(0.5, 0.5) = ∫∫ 2 ds dt over the region: 0 ≤ s ≤ t ≤ 0.5
"""
        },
        {
            "description": "Set up the limits of integration",
            "content": """
Since we need 0 ≤ s ≤ t ≤ 0.5, and we also need s ≤ 0.5 and t ≤ 0.5:
- For a given t in [0, 0.5], s ranges from 0 to t
- t ranges from 0 to 0.5

So: F(0.5, 0.5) = ∫[0 to 0.5] ∫[0 to t] 2 ds dt
"""
        },
        {
            "description": "Integrate with respect to s",
            "content": """
First integrate with respect to s:
∫[0 to t] 2 ds = 2s |[0 to t] = 2t - 0 = 2t

So we have: F(0.5, 0.5) = ∫[0 to 0.5] 2t dt
"""
        },
        {
            "description": "Integrate with respect to t and evaluate",
            "content": """
Now integrate with respect to t:
∫[0 to 0.5] 2t dt = t² |[0 to 0.5] = (0.5)² - 0² = 0.25

Therefore: F(0.5, 0.5) = 0.25
"""
        }
    ]
    
    # Initialize critic
    critic = Critic(model_name="gpt-4o", temperature=0.3)
    
    print(f"\nProblem: {problem}")
    print("\nSolution Steps:")
    for i, step in enumerate(correct_steps):
        print(f"\n  Step {i+1}: {step['description']}")
        print(f"    {step['content'][:100]}...")
    
    print("\n" + "=" * 80)
    print("CRITIQUING CORRECT SOLUTION")
    print("=" * 80)
    
    # Critique the last step (which has the calculation)
    # We need to get the full state to see tool calls
    initial_state = {
        "math_problem": problem,
        "solution_steps": correct_steps[:4],  # Include steps up to target
        "target_step_index": 3,
        "messages": [],
        "critique": None,
    }
    
    result_state = critic.graph.invoke(initial_state)
    critique = result_state["critique"]
    messages = result_state.get("messages", [])
    
    # Check for tool calls in messages
    print("\n🔧 Tool Call Analysis:")
    tool_call_count = 0
    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_count += len(msg.tool_calls)
            print(f"  Message {i}: {len(msg.tool_calls)} tool call(s)")
            for tc in msg.tool_calls:
                print(f"    - {tc['name']}({tc.get('args', {})})")
        elif hasattr(msg, "name") and msg.name:
            # This is a tool response
            print(f"  Message {i}: Tool response from '{msg.name}'")
    
    if tool_call_count > 0:
        print(f"  ✓ Total tool calls: {tool_call_count}")
    else:
        print(f"  ⚠ No tool calls detected")
    
    print(f"\n📋 Step {critique['step_number']}: {critique['step_description']}")
    print(f"\n✓ Logic Correct: {critique['is_logically_correct']}")
    print(f"  {critique['logic_feedback']}")
    print(f"\n🔢 Calculation Correct: {critique['is_calculation_correct']}")
    print(f"  {critique['calculation_feedback']}")
    print(f"\n🏷️  Knowledge Points: {', '.join(critique['knowledge_points'])}")
    
    return critique


def test_incorrect_solution():
    """Test critic agent with an incorrect solution."""
    
    print("\n\n" + "=" * 80)
    print("TEST 2: Incorrect Solution (Wrong Integration Limits)")
    print("=" * 80)
    
    problem = """
Given the joint probability density function:
f(x,y) = 2 for 0 ≤ x ≤ y ≤ 1

Find the joint cumulative distribution function F(x,y) at the point (0.5, 0.5).
"""
    
    # NOTE: This solution has errors - the integration limits ignore the constraint s ≤ t
    # The critic should detect that the limits are wrong and the final answer is incorrect
    incorrect_steps = [
        {
            "description": "Set up the double integral for F(0.5, 0.5)",
            "content": """
The joint cumulative distribution function is defined as:
F(x,y) = ∫∫ f(s,t) ds dt, where the integration is over the region where s ≤ x and t ≤ y.

For F(0.5, 0.5), we need:
F(0.5, 0.5) = ∫∫ 2 ds dt over the region: 0 ≤ s ≤ t ≤ 0.5
"""
        },
        {
            "description": "Set up the limits of integration",  # Error here: wrong limits
            "content": """
For the region 0 ≤ s ≤ 0.5 and 0 ≤ t ≤ 0.5:
F(0.5, 0.5) = ∫[0 to 0.5] ∫[0 to 0.5] 2 ds dt
"""
        },
        {
            "description": "Integrate with respect to s",
            "content": """
∫[0 to 0.5] 2 ds = 2s |[0 to 0.5] = 2(0.5) - 0 = 1

So we have: F(0.5, 0.5) = ∫[0 to 0.5] 1 dt
"""
        },
        {
            "description": "Calculate final result",
            "content": """
∫[0 to 0.5] 1 dt = t |[0 to 0.5] = 0.5 - 0 = 0.5

Therefore: F(0.5, 0.5) = 0.5
"""
        }
    ]
    
    # Initialize critic
    critic = Critic(model_name="gpt-4o", temperature=0.3)
    
    print(f"\nProblem: {problem}")
    print("\nSolution Steps (with intentional error):")
    for i, step in enumerate(incorrect_steps):
        print(f"\n  Step {i+1}: {step['description']}")
        print(f"    {step['content'][:100]}...")
    
    print("\n" + "=" * 80)
    print("CRITIQUING INCORRECT SOLUTION")
    print("=" * 80)
    
    # Critique step 2 (wrong integration limits)
    initial_state_2 = {
        "math_problem": problem,
        "solution_steps": incorrect_steps[:2],
        "target_step_index": 1,
        "messages": [],
        "critique": None,
    }
    
    result_state_2 = critic.graph.invoke(initial_state_2)
    critique_step2 = result_state_2["critique"]
    messages_2 = result_state_2.get("messages", [])
    
    # Check for tool calls
    print("\n🔧 Tool Call Analysis (Step 2):")
    tool_call_count_2 = sum(len(msg.tool_calls) if hasattr(msg, "tool_calls") and msg.tool_calls else 0 for msg in messages_2)
    if tool_call_count_2 > 0:
        print(f"  ✓ Total tool calls: {tool_call_count_2}")
        for i, msg in enumerate(messages_2):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    - {tc['name']}")
    else:
        print(f"  ⚠ No tool calls detected")
    
    print(f"\n📋 Step {critique_step2['step_number']}: {critique_step2['step_description']}")
    print(f"\n✓ Logic Correct: {critique_step2['is_logically_correct']}")
    print(f"  {critique_step2['logic_feedback']}")
    print(f"\n🔢 Calculation Correct: {critique_step2['is_calculation_correct']}")
    print(f"  {critique_step2['calculation_feedback']}")
    print(f"\n🏷️  Knowledge Points: {', '.join(critique_step2['knowledge_points'])}")
    
    # Also critique the final step
    print("\n" + "-" * 80)
    print("CRITIQUING FINAL CALCULATION")
    print("-" * 80)
    
    initial_state_final = {
        "math_problem": problem,
        "solution_steps": incorrect_steps,
        "target_step_index": 3,
        "messages": [],
        "critique": None,
    }
    
    result_state_final = critic.graph.invoke(initial_state_final)
    critique_final = result_state_final["critique"]
    messages_final = result_state_final.get("messages", [])
    
    # Check for tool calls
    print("\n🔧 Tool Call Analysis (Final Step):")
    tool_call_count_final = sum(len(msg.tool_calls) if hasattr(msg, "tool_calls") and msg.tool_calls else 0 for msg in messages_final)
    if tool_call_count_final > 0:
        print(f"  ✓ Total tool calls: {tool_call_count_final}")
        for i, msg in enumerate(messages_final):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    - {tc['name']}")
    else:
        print(f"  ⚠ No tool calls detected")
    
    print(f"\n📋 Step {critique_final['step_number']}: {critique_final['step_description']}")
    print(f"\n✓ Logic Correct: {critique_final['is_logically_correct']}")
    print(f"  {critique_final['logic_feedback']}")
    print(f"\n🔢 Calculation Correct: {critique_final['is_calculation_correct']}")
    print(f"  {critique_final['calculation_feedback']}")
    print(f"\n🏷️  Knowledge Points: {', '.join(critique_final['knowledge_points'])}")
    
    return critique_step2, critique_final


def test_simple_calculation():
    """Test with a complex calculation containing an error to verify tool usage."""
    
    print("\n\n" + "=" * 80)
    print("TEST 3: Complex Calculation Test (With Error)")
    print("=" * 80)
    
    problem = """
Calculate the derivative of f(x) = x³ + 2x² - 5x + 3 at x = 2, 
then evaluate: f'(2) × π + e²
"""
    
    # Note: Intentional error in the calculation
    # Correct: f'(x) = 3x² + 4x - 5, f'(2) = 3(4) + 4(2) - 5 = 12 + 8 - 5 = 15
    # f'(2) × π + e² = 15π + e² ≈ 47.14 + 7.39 ≈ 54.53
    # But we'll provide wrong intermediate calculation (claiming f'(2) = 17 instead of 15)
    complex_steps = [
        {
            "description": "Find the derivative f'(x)",
            "content": """
Using the power rule:
f(x) = x³ + 2x² - 5x + 3
f'(x) = 3x² + 4x - 5
"""
        },
        {
            "description": "Evaluate f'(2)",
            "content": """
Substitute x = 2 into f'(x):
f'(2) = 3(2)² + 4(2) - 5
f'(2) = 3(4) + 8 - 5
f'(2) = 12 + 8 - 5
f'(2) = 17
"""
        },
        {
            "description": "Calculate final result",
            "content": """
Now compute: f'(2) × π + e²
= 17 × π + e²
= 17π + e²
≈ 53.41 + 7.39
≈ 60.80
"""
        }
    ]
    
    critic = Critic(model_name="gpt-4o", temperature=0.3)
    
    print(f"\nProblem: {problem}")
    print("\nSolution Steps (with intentional error in step 2):")
    for i, step in enumerate(complex_steps):
        print(f"\n  Step {i+1}: {step['description']}")
        if i == 1:
            print(f"    ⚠️  ERROR: Claims f'(2) = 17, but should be 15")
    
    # Get full state to see tool calls - critique step 2 with the error
    initial_state_simple = {
        "math_problem": problem,
        "solution_steps": complex_steps[:2],
        "target_step_index": 1,  # Check the step with the error
        "messages": [],
        "critique": None,
    }
    
    result_state_simple = critic.graph.invoke(initial_state_simple)
    critique = result_state_simple["critique"]
    messages_simple = result_state_simple.get("messages", [])
    
    # Check for tool calls in detail
    print("\n🔧 Tool Call Analysis:")
    tool_call_count = 0
    for i, msg in enumerate(messages_simple):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_count += len(msg.tool_calls)
            print(f"  Message {i}: {len(msg.tool_calls)} tool call(s)")
            for tc in msg.tool_calls:
                print(f"    - Tool: {tc['name']}")
                args_str = str(tc.get('args', {}))
                if len(args_str) > 80:
                    args_str = args_str[:77] + "..."
                print(f"      Args: {args_str}")
        elif hasattr(msg, "name") and msg.name:
            # This is a tool response
            print(f"  Message {i}: Tool response from '{msg.name}'")
            content_str = str(msg.content)
            if len(content_str) > 100:
                content_str = content_str[:97] + "..."
            print(f"      Content: {content_str}")
    
    if tool_call_count > 0:
        print(f"\n  ✓ Total tool calls: {tool_call_count}")
    else:
        print(f"\n  ⚠ No tool calls detected")
    
    print(f"\n📋 Critique:")
    print(f"  Logic Correct: {critique['is_logically_correct']}")
    print(f"  Calculation Correct: {critique['is_calculation_correct']}")
    print(f"  Logic Feedback: {critique['logic_feedback']}")
    print(f"  Calculation Feedback: {critique['calculation_feedback']}")
    print(f"  Knowledge Points: {', '.join(critique['knowledge_points'])}")
    
    # Check if error was detected
    if not critique['is_calculation_correct']:
        print("\n  ✅ Agent successfully detected the calculation error!")
    else:
        print("\n  ❌ Agent failed to detect the calculation error")
    
    return critique


def test_tool_result_accessibility():
    """Test that the agent can access tool call results (both success and errors) when making critique."""
    
    print("\n" + "=" * 80)
    print("TEST 4: Tool Result Accessibility")
    print("=" * 80)
    
    critic = Critic(temperature=0.3)
    
    problem = """
Verify if the following calculation is correct:
sqrt(16) + sqrt(25) = 9
"""
    
    steps = [
        {
            "description": "Calculate square roots and sum",
            "content": """
We need to evaluate:
sqrt(16) + sqrt(25)

Breaking it down:
- sqrt(16) = 4
- sqrt(25) = 5
- Therefore: 4 + 5 = 9

The calculation is correct.
"""
        }
    ]
    
    critique, token_usage = critic.critique_step(problem, steps, 0)
    
    print(f"\n📋 Critique Result:")
    print(f"  Calculation Correct: {critique['is_calculation_correct']}")
    print(f"  Calculation Feedback:\n  {critique['calculation_feedback']}")
    print(f"  Token Usage: {token_usage['input_tokens']} input, {token_usage['output_tokens']} output")
    
    # Check the messages to see if tool results are present
    print(f"\n🔍 Checking Message History:")
    if hasattr(critic, 'graph'):
        # Try to get the last execution state
        print("  Messages from last execution:")
        # The messages should include ToolMessage objects with results
        
    # The feedback should reference the tool results
    feedback_lower = critique['calculation_feedback'].lower()
    has_tool_reference = any(keyword in feedback_lower for keyword in 
                             ['result', 'tool', 'verify', 'calculation', '4', '5', '9'])
    
    print(f"\n📊 Tool Result Accessibility:")
    print(f"  Feedback references calculations: {'✅ YES' if has_tool_reference else '❌ NO'}")
    print(f"  Step is correct: {'✅ YES' if critique['is_calculation_correct'] else '❌ NO'}")
    print(f"  Feedback is concise (for correct result): {'✅ YES' if len(critique['calculation_feedback']) < 50 else '⚠️ Verbose'}")
    
    return critique


def test_tool_error_accessibility():
    """Test that the agent can access tool errors when tools fail."""
    
    print("\n" + "=" * 80)
    print("TEST 5: Tool Error Accessibility")
    print("=" * 80)
    
    critic = Critic(temperature=0.3)
    
    problem = """
Evaluate the expression with an intentionally problematic formula:
1 / 0
"""
    
    steps = [
        {
            "description": "Attempt to calculate 1/0",
            "content": """
Calculating: 1 / 0 = infinity (or undefined)
"""
        }
    ]
    
    critique, token_usage = critic.critique_step(problem, steps, 0)
    
    print(f"\n📋 Critique Result:")
    print(f"  Logic Correct: {critique['is_logically_correct']}")
    print(f"  Calculation Correct: {critique['is_calculation_correct']}")
    print(f"  Feedback:\n  {critique['calculation_feedback']}")
    print(f"  Token Usage: {token_usage['input_tokens']} input, {token_usage['output_tokens']} output")
    
    # Check if the feedback mentions the division by zero or error
    feedback_lower = critique['calculation_feedback'].lower()
    mentions_error = any(keyword in feedback_lower for keyword in 
                        ['error', 'undefined', 'division', 'zero', 'invalid'])
    
    print(f"\n📊 Error Handling:")
    print(f"  Feedback mentions error/issue: {'✅ YES' if mentions_error else '❌ NO'}")
    print(f"  Step marked as incorrect: {'✅ YES' if not critique['is_calculation_correct'] else '❌ NO'}")
    
    return critique


if __name__ == "__main__":
    print("\n🧪 Testing Critic Agent with Calculator Tools\n")
    
    try:
        # Test 1: Correct solution
        critique1 = test_correct_solution()
        
        # Test 2: Incorrect solution
        critique2_step2, critique2_final = test_incorrect_solution()
        
        # Test 3: Simple calculation
        critique3 = test_simple_calculation()
        
        # Test 4: Tool result accessibility
        critique4 = test_tool_result_accessibility()
        
        # Test 5: Tool error accessibility
        critique5 = test_tool_error_accessibility()
        
        print("\n\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nSummary:")
        print(f"  Test 1 (Correct): Logic={critique1['is_logically_correct']}, Calc={critique1['is_calculation_correct']}")
        print(f"  Test 2 (Incorrect Step 2): Logic={critique2_step2['is_logically_correct']}, Calc={critique2_step2['is_calculation_correct']}")
        print(f"  Test 2 (Incorrect Final): Logic={critique2_final['is_logically_correct']}, Calc={critique2_final['is_calculation_correct']}")
        print(f"  Test 3 (Complex w/ Error): Logic={critique3['is_logically_correct']}, Calc={critique3['is_calculation_correct']}")
        print(f"  Test 4 (Tool Access): Calc={critique4['is_calculation_correct']}, Concise={'Yes' if len(critique4['calculation_feedback']) < 50 else 'No'}")
        print(f"  Test 5 (Error Access): Logic={critique5['is_logically_correct']}, Calc={critique5['is_calculation_correct']}")
        
        # Expected outcomes
        print("\n📊 Expected vs Actual:")
        print(f"  Test 3 should detect error in f'(2) calculation (17 vs 15): {'✅ PASS' if not critique3['is_calculation_correct'] else '❌ FAIL'}")
        print(f"  Test 4 should give concise feedback for correct step: {'✅ PASS' if len(critique4['calculation_feedback']) < 50 else '❌ FAIL'}")
        print(f"  Test 5 should detect division by zero issue: {'✅ PASS' if not critique5['is_calculation_correct'] else '❌ FAIL'}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
