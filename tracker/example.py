"""
Example usage of the tracker decorator.
"""

from tracker import tracker


# Example 1: Basic usage
@tracker(name="llm_gen", value={
    "query": 'args[0].get("user_query")',
    "answer": 'ret[0].get("llm_response")'
})
def get_llm_generation(state):
    """Simulated LLM generation function."""
    # Simulated LLM generation
    llm_response = f"Response to: {state.get('user_query')}"
    return {
        "llm_response": llm_response,
        "tokens": 150
    }


# Example 2: Tracking with multiple arguments
@tracker(name="data_processor", value={
    "input_text": 'args[0]',
    "config": 'args[1]',
    "result": 'ret[0]'
})
def process_data(text, config):
    """Process data with configuration."""
    processed = text.upper() if config.get("uppercase") else text.lower()
    return {"processed": processed, "length": len(processed)}


# Example 3: Tracking with error handling
@tracker(name="division", value={
    "numerator": 'args[0]',
    "denominator": 'args[1]',
    "result": 'ret[0]'
})
def divide(a, b):
    """Divide two numbers."""
    return a / b


if __name__ == "__main__":
    from tracker import Tracker
    
    print("Running tracker examples...")
    
    # Example 1
    print("\n1. LLM Generation:")
    state = {"user_query": "What is the meaning of life?"}
    result1 = get_llm_generation(state)
    print(f"Result: {result1}")
    
    # Example 2
    print("\n2. Data Processing:")
    result2 = process_data("Hello World", {"uppercase": True})
    print(f"Result: {result2}")
    
    # Example 3 - Success
    print("\n3. Division (success):")
    result3 = divide(10, 2)
    print(f"Result: {result3}")
    
    # Example 4 - Error case
    print("\n4. Division (error):")
    try:
        result4 = divide(10, 0)
    except ZeroDivisionError as e:
        print(f"Caught error: {e}")
    
    # Query records
    print("\n5. Querying records from database:")
    tracker_instance = Tracker()
    records = tracker_instance.get_records(limit=5)
    print(f"Found {len(records)} records")
    for record in records:
        print(f"  - {record['name']} at {record['timestamp']}: {record['status']}")
    
    print("\nData stored in ./data/tracker/record.db")
