"""
Test suite for the MathCalculator class with LaTeX support
"""

from calculator import MathCalculator


def test_numerical_evaluation():
    """Test numerical expression evaluation"""
    calc = MathCalculator()
    
    print("Testing numerical evaluation...")
    
    # Simple baseline
    result = calc.evaluate_numerical("2 + 2")
    print(f"  2 + 2 = {result}")
    assert result == 4.0
    
    # Complex nested fractions with variables
    result = calc.evaluate_numerical(r"\frac{2x + 3}{x - 1} + \frac{x}{2}", {"x": 5})
    print(f"  (2x+3)/(x-1) + x/2 [x=5] = {result}")
    assert abs(result - 5.75) < 1e-10
    
    # Trigonometric composition with multiple constants
    result = calc.evaluate_numerical(r"\sin^{2}\left(\tfrac{\pi}{4}\right) + \cos^{2}\left(\tfrac{\pi}{4}\right)")
    print(f"  sin²(π/4) + cos²(π/4) = {result}")
    assert abs(result - 1.0) < 1e-10
    
    # Definite integral evaluation
    result = calc.evaluate_numerical(r"\int_{0}^{\pi} \sin(x) \, dx")
    print(f"  ∫₀^π sin(x)dx = {result}")
    assert abs(result - 2.0) < 1e-6
    
    print("✓ Numerical evaluation tests passed\n")


def test_symbolic_evaluation():
    """Test symbolic expression evaluation"""
    calc = MathCalculator()
    
    print("Testing symbolic evaluation...")
    
    # Simple baseline
    result = calc.evaluate_symbolic("x + 1")
    print(f"  x + 1 = {result}")
    
    # Complex polynomial factorization
    result = calc.evaluate_symbolic(r"x^{3} - 3x^{2} + 3x - 1", simplify=True)
    print(f"  x³ - 3x² + 3x - 1 simplified = {result}")
    
    # Rational function simplification
    result = calc.evaluate_symbolic(r"\frac{x^{3} + x^{2} - x - 1}{x^{2} - 1}", simplify=True)
    print(f"  (x³+x²-x-1)/(x²-1) simplified = {result}")
    
    # Nested fraction with substitution
    result = calc.evaluate_symbolic(r"\frac{1}{1 + \frac{1}{x}}", {"x": 3})
    print(f"  1/(1+1/x) [x=3] = {result}")
    
    print("✓ Symbolic evaluation tests passed\n")


def test_verify_calculation():
    """Test calculation verification"""
    calc = MathCalculator()
    
    print("Testing calculation verification...")
    
    # Simple baseline
    is_correct, msg = calc.verify_calculation("2 + 2", 4)
    print(f"  {msg}")
    assert is_correct
    
    # Complex trigonometric identity
    is_correct, msg = calc.verify_calculation(
        r"\sin(2x)", 
        r"2\sin(x)\cos(x)"
    )
    print(f"  {msg}")
    assert is_correct
    
    # Nested fraction computation
    is_correct, msg = calc.verify_calculation(
        r"\frac{\tfrac{1}{2} + \tfrac{2}{3}}{\tfrac{3}{4}}",
        7/6 / (3/4)
    )
    print(f"  {msg}")
    assert is_correct
    
    # Polynomial expansion equivalence
    is_correct, msg = calc.verify_calculation(
        r"(x+1)(x+2)(x+3)", 
        r"x^{3} + 6x^{2} + 11x + 6"
    )
    print(f"  {msg}")
    assert is_correct
    
    # Exponential properties
    is_correct, msg = calc.verify_calculation(
        r"e^{x + y}", 
        r"e^{x} \cdot e^{y}"
    )
    print(f"  {msg}")
    assert is_correct
    
    # Incorrect calculation for testing
    is_correct, msg = calc.verify_calculation(
        r"\sin^{2}(x) + \cos^{2}(x)", 
        r"2"
    )
    print(f"  {msg}")
    assert not is_correct
    
    print("✓ Verification tests passed\n")


def test_compare_expressions():
    """Test expression comparison"""
    calc = MathCalculator()
    
    print("Testing expression comparison...")
    
    # Simple baseline
    are_equal, msg = calc.compare_expressions("x + 1", "1 + x")
    print(f"  {msg}")
    assert are_equal
    
    # Complex polynomial factorization equivalence
    are_equal, msg = calc.compare_expressions(
        r"x^{4} - 1", 
        r"(x^{2} + 1)(x + 1)(x - 1)"
    )
    print(f"  {msg}")
    assert are_equal
    
    # Rational function equivalence after simplification
    are_equal, msg = calc.compare_expressions(
        r"\frac{x^{2} - 4x + 4}{x - 2}", 
        r"x - 2"
    )
    print(f"  {msg}")
    assert are_equal
    
    # Trigonometric identity
    are_equal, msg = calc.compare_expressions(
        r"\tan^{2}(x) + 1", 
        r"\sec^{2}(x)"
    )
    print(f"  {msg}")
    assert are_equal
    
    # Non-equivalent expressions
    are_equal, msg = calc.compare_expressions(
        r"x^{3} + x", 
        r"x^{3} + x^{2}"
    )
    print(f"  {msg}")
    assert not are_equal
    
    print("✓ Comparison tests passed\n")


def test_complex_expressions():
    """Test complex mathematical expressions"""
    calc = MathCalculator()
    
    print("Testing complex expressions...")
    
    # Simple baseline
    result = calc.evaluate_numerical("e**2")
    print(f"  e² = {result}")
    assert abs(result - 7.389) < 0.001
    
    # Nested trigonometric and exponential
    result = calc.evaluate_numerical(r"e^{\sin(\pi/2)} + \ln(e^{3})")
    print(f"  e^sin(π/2) + ln(e³) = {result}")
    assert abs(result - (2.71828 + 3)) < 0.01
    
    # Complex fraction with multiple operations
    result = calc.evaluate_numerical(
        r"\frac{\sqrt{x^{2} + y^{2}}}{\sin(\pi/6) + \cos(\pi/3)}",
        {"x": 3, "y": 4}
    )
    print(f"  √(x²+y²)/(sin(π/6)+cos(π/3)) [x=3,y=4] = {result}")
    assert abs(result - 5.0) < 1e-6
    
    # Double integral (evaluated as nested)
    result = calc.evaluate_numerical(r"\int_{0}^{1} x^{3} \, dx")
    print(f"  ∫₀¹ x³ dx = {result}")
    assert abs(result - 0.25) < 1e-8
    
    print("✓ Complex expression tests passed\n")


def test_advanced_mathematical_operations():
    """Advanced mathematical operations: series, products, limits"""
    calc = MathCalculator()

    print("Testing advanced mathematical operations...")

    # Compound fraction arithmetic
    res = calc.evaluate_numerical(r"\frac{\tfrac{3}{4} + \tfrac{5}{6}}{\tfrac{7}{8}}")
    print(f"  (3/4 + 5/6)/(7/8) = {res}")
    assert abs(res - ((3/4 + 5/6)/(7/8))) < 1e-10

    # Complex definite integral with trig
    res = calc.evaluate_numerical(r"\int_{0}^{\pi/2} \sin(x) + \cos(x) \, dx")
    print(f"  ∫₀^(π/2) (sin(x)+cos(x))dx = {res}")
    assert abs(res - 2.0) < 1e-6

    # Polynomial identity with high degree
    eq, msg = calc.compare_expressions(
        r"(x+1)^{4}",
        r"x^{4} + 4x^{3} + 6x^{2} + 4x + 1"
    )
    print(f"  {msg}")
    assert eq

    # Complex algebraic simplification
    ok, msg = calc.verify_calculation(
        r"\frac{x^{4} - y^{4}}{x^{2} - y^{2}}",
        r"x^{2} + y^{2}"
    )
    print(f"  {msg}")
    assert ok

    print("✓ Advanced mathematical operations tests passed\n")


if __name__ == "__main__":
    test_numerical_evaluation()
    test_symbolic_evaluation()
    test_verify_calculation()
    test_compare_expressions()
    test_complex_expressions()
    test_advanced_mathematical_operations()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
