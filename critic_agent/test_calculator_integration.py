"""
Test calculator integration with critic agent.
Verify that the agent can:
1. Make multiple tool calls to verify complex calculations
2. Split complex expressions that don't fit into one verification
3. Handle LaTeX expressions properly
"""

from dotenv import load_dotenv
from critic_agent.critic import Critic

load_dotenv()


def test_multiple_verification_steps():
    """Test that critic can make multiple tool calls to verify a complex calculation."""
    
    print("=" * 80)
    print("TEST 1: Multiple Verification Steps")
    print("=" * 80)
    
    problem = """
Calculate the value of the expression:
\\[
\\frac{\\sin^2(\\pi/4) + \\cos^2(\\pi/4)}{\\sqrt{16}} + \\int_0^1 x^2 \\, dx
\\]
"""
    
    # Solution with multiple sub-calculations that should be verified separately
    steps = [
        {
            "description": "Break down and evaluate each component",
            "content": r"""
We need to evaluate three components:

1. Trigonometric identity: \(\sin^2(\pi/4) + \cos^2(\pi/4) = 1\)
2. Square root: \(\sqrt{16} = 4\)
3. Definite integral: \(\int_0^1 x^2 \, dx = \frac{1}{3}\)

So the expression becomes:
\[
\frac{1}{4} + \frac{1}{3} = \frac{3}{12} + \frac{4}{12} = \frac{7}{12}
\]

Therefore, the final answer is \(\frac{7}{12}\).
"""
        }
    ]
    
    critic = Critic(model_name="gpt-4o", temperature=0.3)
    
    print(f"\nProblem: {problem}")
    print(f"\nStep: {steps[0]['description']}")
    print(f"Content preview: {steps[0]['content'][:150]}...")
    
    print("\n" + "=" * 80)
    print("CRITIQUING WITH MULTIPLE VERIFICATIONS")
    print("=" * 80)
    
    initial_state = {
        "math_problem": problem,
        "solution_steps": steps,
        "target_step_index": 0,
        "messages": [],
        "critique": None,
    }
    
    result_state = critic.graph.invoke(initial_state)
    critique = result_state["critique"]
    messages = result_state.get("messages", [])
    
    # Analyze tool calls
    print("\n🔧 Tool Call Analysis:")
    tool_call_count = 0
    tool_names = []
    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_count += len(msg.tool_calls)
            for tc in msg.tool_calls:
                tool_name = tc.get('name', 'unknown')
                tool_names.append(tool_name)
                print(f"  Message {i}: Tool '{tool_name}'")
                args_str = str(tc.get('args', {}))
                if len(args_str) > 100:
                    args_str = args_str[:97] + "..."
                print(f"    Args: {args_str}")
        elif hasattr(msg, "name") and msg.name:
            print(f"  Message {i}: Tool response from '{msg.name}'")
            content_str = str(msg.content)[:150]
            print(f"    Result: {content_str}")
    
    print(f"\n  Total tool calls: {tool_call_count}")
    print(f"  Tools used: {', '.join(set(tool_names))}")
    
    print(f"\n📋 Critique Results:")
    print(f"  Logic Correct: {critique['is_logically_correct']}")
    print(f"  Calculation Correct: {critique['is_calculation_correct']}")
    print(f"  Logic Feedback: {critique['logic_feedback'][:200] if critique['logic_feedback'] else 'N/A'}...")
    print(f"  Calculation Feedback: {critique['calculation_feedback'][:200] if critique['calculation_feedback'] else 'N/A'}...")
    
    # Check if multiple tools were called
    if tool_call_count >= 3:
        print(f"\n  ✅ Agent made multiple tool calls ({tool_call_count}) as expected")
    else:
        print(f"\n  ⚠️  Agent made fewer tool calls ({tool_call_count}) than expected")
    
    return critique, tool_call_count


def test_expression_splitting():
    """Test that critic can split complex expressions that don't fit one verification."""
    
    print("\n\n" + "=" * 80)
    print("TEST 2: Expression Splitting for Unsupported Operations")
    print("=" * 80)
    
    problem = """
Evaluate the following expression:
\\[
E = \\int_0^1 x^2 \\, dx + \\int_0^{\\pi} \\sin(x) \\, dx
\\]
"""
    
    # Solution that correctly evaluates each integral separately
    steps = [
        {
            "description": "Evaluate each integral separately then sum",
            "content": r"""
Since the expression contains two separate integrals, we evaluate them individually:

**First integral:**
\[
\int_0^1 x^2 \, dx = \left[\frac{x^3}{3}\right]_0^1 = \frac{1}{3} - 0 = \frac{1}{3}
\]

**Second integral:**
\[
\int_0^{\pi} \sin(x) \, dx = [-\cos(x)]_0^{\pi} = -\cos(\pi) - (-\cos(0)) = -(-1) - (-1) = 1 + 1 = 2
\]

**Final sum:**
\[
E = \frac{1}{3} + 2 = \frac{1}{3} + \frac{6}{3} = \frac{7}{3} \approx 2.333
\]
"""
        }
    ]
    
    critic = Critic(model_name="gpt-4o", temperature=0.3)
    
    print(f"\nProblem: {problem}")
    print(f"\nStep: {steps[0]['description']}")
    
    print("\n" + "=" * 80)
    print("CRITIQUING WITH EXPRESSION SPLITTING")
    print("=" * 80)
    
    initial_state = {
        "math_problem": problem,
        "solution_steps": steps,
        "target_step_index": 0,
        "messages": [],
        "critique": None,
    }
    
    result_state = critic.graph.invoke(initial_state)
    critique = result_state["critique"]
    messages = result_state.get("messages", [])
    
    # Analyze tool calls
    print("\n🔧 Tool Call Analysis:")
    tool_call_count = 0
    integral_checks = 0
    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_count += 1
                tool_name = tc.get('name', 'unknown')
                args = tc.get('args', {})
                expression = args.get('expression', '') or args.get('expr1', '')
                
                # Check if verifying integrals separately
                if 'int_' in expression or '\\int' in expression:
                    integral_checks += 1
                    print(f"  Tool call {tool_call_count}: {tool_name}")
                    print(f"    Checking integral: {expression[:80]}...")
    
    print(f"\n  Total tool calls: {tool_call_count}")
    print(f"  Integral verifications: {integral_checks}")
    
    print(f"\n📋 Critique Results:")
    print(f"  Logic Correct: {critique['is_logically_correct']}")
    print(f"  Calculation Correct: {critique['is_calculation_correct']}")
    
    # Check if agent properly split the verification
    if integral_checks >= 2:
        print(f"\n  ✅ Agent correctly verified integrals separately ({integral_checks} checks)")
    else:
        print(f"\n  ℹ️  Agent made {integral_checks} integral verification(s)")
    
    return critique, integral_checks


def test_latex_complex_expression():
    """Test critic with complex LaTeX expressions requiring multiple verifications."""
    
    print("\n\n" + "=" * 80)
    print("TEST 3: Complex LaTeX Expression with Multiple Components")
    print("=" * 80)
    
    problem = """
Simplify and evaluate:
\\[
\\frac{x^4 - 1}{x^2 - 1} \\text{ at } x = 3
\\]
"""
    
    steps = [
        {
            "description": "Factor and simplify, then evaluate",
            "content": r"""
First, we factor the numerator and denominator:

**Numerator:** \(x^4 - 1 = (x^2 + 1)(x^2 - 1)\)

**Denominator:** \(x^2 - 1 = (x+1)(x-1)\)

So we have:
\[
\frac{x^4 - 1}{x^2 - 1} = \frac{(x^2 + 1)(x^2 - 1)}{x^2 - 1} = x^2 + 1
\]

Now evaluating at \(x = 3\):
\[
x^2 + 1 = 3^2 + 1 = 9 + 1 = 10
\]

Therefore, the answer is 10.
"""
        }
    ]
    
    critic = Critic(model_name="gpt-4o", temperature=0.3)
    
    print(f"\nProblem: {problem}")
    
    print("\n" + "=" * 80)
    print("CRITIQUING COMPLEX LATEX WITH MULTIPLE STEPS")
    print("=" * 80)
    
    initial_state = {
        "math_problem": problem,
        "solution_steps": steps,
        "target_step_index": 0,
        "messages": [],
        "critique": None,
    }
    
    result_state = critic.graph.invoke(initial_state)
    critique = result_state["critique"]
    messages = result_state.get("messages", [])
    
    # Analyze tool calls
    print("\n🔧 Tool Call Analysis:")
    tool_call_count = 0
    verification_types = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_count += 1
                tool_name = tc.get('name', 'unknown')
                verification_types.append(tool_name)
                print(f"  Tool: {tool_name}")
    
    print(f"\n  Total tool calls: {tool_call_count}")
    print(f"  Verification types: {', '.join(set(verification_types))}")
    
    print(f"\n📋 Critique Results:")
    print(f"  Logic Correct: {critique['is_logically_correct']}")
    print(f"  Calculation Correct: {critique['is_calculation_correct']}")
    
    # Check for multiple verification types
    if len(set(verification_types)) >= 2:
        print(f"\n  ✅ Agent used multiple verification methods")
    
    return critique


if __name__ == "__main__":
    print("\n🧪 Testing Calculator Integration with Critic Agent\n")
    
    try:
        # Test 1: Multiple verifications
        critique1, tool_count1 = test_multiple_verification_steps()
        
        # Test 2: Expression splitting
        critique2, integral_count = test_expression_splitting()
        
        # Test 3: Complex LaTeX
        critique3 = test_latex_complex_expression()
        
        print("\n\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nSummary:")
        print(f"  Test 1: {tool_count1} tool calls (expected >= 3)")
        print(f"  Test 2: {integral_count} integral verifications (expected >= 2)")
        print(f"  Test 3: Complex LaTeX verification completed")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
