"""
Test to check if tool call tokens are being tracked.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from critic_agent.critic import Critic

def test_tool_call_token_tracking():
    """Test token tracking when tools are called."""
    
    # Initialize critic
    critic = Critic(model_name="gpt-4o-mini-2024-07-18", temperature=0.3)
    
    # Problem that should trigger tool calls (complex calculation)
    problem = "Verify the following integral calculation"
    steps = [
        {
            "description": "Calculate integral",
            "content": r"""
Let's calculate: ∫₀^π sin(x) dx

Using the antiderivative:
-cos(x) |₀^π = -cos(π) - (-cos(0)) = -(-1) - (-1) = 1 + 1 = 2
"""
        }
    ]
    
    print("Testing critique with tool calls...")
    print("="*80)
    
    # Run critique
    critique, token_usage = critic.critique_step(problem, steps, 0)
    
    print(f"\n✓ Critique completed")
    print(f"  Logic Correct: {critique['is_logically_correct']}")
    print(f"  Calculation Correct: {critique['is_calculation_correct']}")
    
    # Check the actual messages to see what's tracked
    print("\n" + "="*80)
    print("Analyzing message metadata...")
    print("="*80)
    
    # Re-run to inspect messages
    from critic_agent.state import CriticState
    initial_state: CriticState = {
        "math_problem": problem,
        "solution_steps": steps,
        "target_step_index": 0,
        "messages": [],
        "critique": None,
    }
    
    result = critic.graph.invoke(initial_state)
    messages = result.get("messages", [])
    
    tool_call_count = 0
    ai_message_count = 0
    
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        print(f"\nMessage {i}: {msg_type}")
        
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_count += len(msg.tool_calls)
            print(f"  Tool calls: {len(msg.tool_calls)}")
            for tc in msg.tool_calls:
                print(f"    - {tc.get('name', 'unknown')}")
        
        if msg_type == "AIMessage":
            ai_message_count += 1
            
            # Check for usage_metadata
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                print(f"  usage_metadata: {msg.usage_metadata}")
                
            # Check for response_metadata
            if hasattr(msg, "response_metadata") and msg.response_metadata:
                metadata = msg.response_metadata
                print(f"  response_metadata keys: {list(metadata.keys())}")
                
                if "token_usage" in metadata:
                    print(f"    token_usage: {metadata['token_usage']}")
                    
            # Check content preview
            content_preview = str(msg.content)[:100] if msg.content else "None"
            print(f"  content preview: {content_preview}...")
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Total messages: {len(messages)}")
    print(f"AI messages: {ai_message_count}")
    print(f"Tool calls: {tool_call_count}")
    print(f"\n📊 Tracked Token Usage:")
    print(f"  Input Tokens:  {token_usage['input_tokens']}")
    print(f"  Output Tokens: {token_usage['output_tokens']}")
    print(f"  Total Tokens:  {token_usage['input_tokens'] + token_usage['output_tokens']}")
    
    if tool_call_count > 0:
        print(f"\n✅ Tools were called - checking if their tokens are included...")
        print(f"   Note: The token counts above should include tokens for:")
        print(f"   - The initial prompt")
        print(f"   - Tool call requests (function names/args)")
        print(f"   - Tool responses")
        print(f"   - Final response generation")
    else:
        print(f"\n⚠️  No tools were called in this test")

if __name__ == "__main__":
    test_tool_call_token_tracking()
