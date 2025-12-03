"""
Quick test to verify token tracking is working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from critic_agent.critic import Critic

def test_token_tracking():
    """Test that token usage is tracked correctly."""
    
    # Initialize critic
    critic = Critic(model_name="gpt-4o-mini-2024-07-18", temperature=0.3)
    
    # Simple test problem
    problem = "What is 2 + 2?"
    steps = [
        {
            "description": "Calculate the sum",
            "content": "2 + 2 = 4"
        }
    ]
    
    print("Testing single step critique with token tracking...")
    critique, token_usage = critic.critique_step(problem, steps, 0)
    
    print(f"\n✓ Critique completed")
    print(f"  Logic Correct: {critique['is_logically_correct']}")
    print(f"  Calculation Correct: {critique['is_calculation_correct']}")
    print(f"\n📊 Token Usage:")
    print(f"  Input Tokens:  {token_usage['input_tokens']}")
    print(f"  Output Tokens: {token_usage['output_tokens']}")
    print(f"  Total Tokens:  {token_usage['input_tokens'] + token_usage['output_tokens']}")
    
    # Test with multiple steps
    print("\n" + "="*80)
    print("Testing multiple steps critique with token tracking...")
    
    problem2 = "Calculate 5 × 3"
    steps2 = [
        {
            "description": "Setup",
            "content": "We need to multiply 5 by 3"
        },
        {
            "description": "Calculate",
            "content": "5 × 3 = 15"
        }
    ]
    
    critiques, total_token_usage = critic.critique_all_steps(problem2, steps2)
    
    print(f"\n✓ All critiques completed")
    print(f"  Number of steps: {len(critiques)}")
    print(f"\n📊 Total Token Usage:")
    print(f"  Input Tokens:  {total_token_usage['input_tokens']}")
    print(f"  Output Tokens: {total_token_usage['output_tokens']}")
    print(f"  Total Tokens:  {total_token_usage['input_tokens'] + total_token_usage['output_tokens']}")
    
    # Verify token usage is greater than 0
    assert token_usage['input_tokens'] > 0, "Input tokens should be > 0"
    assert token_usage['output_tokens'] > 0, "Output tokens should be > 0"
    assert total_token_usage['input_tokens'] > 0, "Total input tokens should be > 0"
    assert total_token_usage['output_tokens'] > 0, "Total output tokens should be > 0"
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_token_tracking()
