"""
Test suite for the simplified MathCalculator class
"""

from calculator import MathCalculator


def test_numerical_evaluation():
    """Test numerical expression evaluation"""
    calc = MathCalculator()
    
    print("Testing numerical evaluation...")
    
    # Basic arithmetic
    result = calc.evaluate_numerical("2 + 2")
    print(f"  2 + 2 = {result}")
    assert result == 4.0
    
    # With variables
    result = calc.evaluate_numerical("2*x + 3", {"x": 5})
    print(f"  2*x + 3 (x=5) = {result}")
    assert result == 13.0
    
    # With constants
    result = calc.evaluate_numerical("sin(pi/2)")
    print(f"  sin(pi/2) = {result}")
    assert abs(result - 1.0) < 1e-6
    
    print("✓ Numerical evaluation tests passed\n")


def test_symbolic_evaluation():
    """Test symbolic expression evaluation"""
    calc = MathCalculator()
    
    print("Testing symbolic evaluation...")
    
    # Basic expression
    result = calc.evaluate_symbolic("x**2 + 2*x + 1")
    print(f"  x**2 + 2*x + 1 = {result}")
    
    # Simplification
    result = calc.evaluate_symbolic("(x + 1)**2", simplify=True)
    print(f"  (x + 1)**2 simplified = {result}")
    
    # With substitution
    result = calc.evaluate_symbolic("x**2 + y", {"x": 2})
    print(f"  x**2 + y (x=2) = {result}")
    
    print("✓ Symbolic evaluation tests passed\n")


def test_verify_calculation():
    """Test calculation verification"""
    calc = MathCalculator()
    
    print("Testing calculation verification...")
    
    # Numeric verification
    is_correct, msg = calc.verify_calculation("2 + 2", 4)
    print(f"  {msg}")
    assert is_correct
    
    # Numeric with variables
    is_correct, msg = calc.verify_calculation("2*x + 3", 13, {"x": 5})
    print(f"  {msg}")
    assert is_correct
    
    # Symbolic verification
    is_correct, msg = calc.verify_calculation("x**2 + 2*x + 1", "(x+1)**2")
    print(f"  {msg}")
    assert is_correct
    
    # Incorrect calculation
    is_correct, msg = calc.verify_calculation("2 + 2", 5)
    print(f"  {msg}")
    assert not is_correct
    
    print("✓ Verification tests passed\n")


def test_compare_expressions():
    """Test expression comparison"""
    calc = MathCalculator()
    
    print("Testing expression comparison...")
    
    # Symbolic equivalence
    are_equal, msg = calc.compare_expressions("x**2 + 2*x + 1", "(x+1)**2")
    print(f"  {msg}")
    assert are_equal
    
    # Different expressions
    are_equal, msg = calc.compare_expressions("x + 1", "x + 2")
    print(f"  {msg}")
    assert not are_equal
    
    # Numeric comparison with variables
    are_equal, msg = calc.compare_expressions("2*x", "x + x", {"x": 5})
    print(f"  {msg}")
    assert are_equal
    
    print("✓ Comparison tests passed\n")


def test_complex_expressions():
    """Test complex mathematical expressions"""
    calc = MathCalculator()
    
    print("Testing complex expressions...")
    
    # Trigonometric
    result = calc.evaluate_numerical("cos(pi) + sin(0)")
    print(f"  cos(pi) + sin(0) = {result}")
    assert abs(result - (-1.0)) < 1e-6
    
    # Exponentials
    result = calc.evaluate_numerical("e**2")
    print(f"  e**2 = {result}")
    assert abs(result - 7.389) < 0.001
    
    # Combined
    result = calc.evaluate_numerical("sqrt(x**2 + y**2)", {"x": 3, "y": 4})
    print(f"  sqrt(3^2 + 4^2) = {result}")
    assert result == 5.0
    
    print("✓ Complex expression tests passed\n")


if __name__ == "__main__":
    test_numerical_evaluation()
    test_symbolic_evaluation()
    test_verify_calculation()
    test_compare_expressions()
    test_complex_expressions()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
