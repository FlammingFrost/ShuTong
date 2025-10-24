"""
Mathematical Calculator Module for Critic Agent

This module provides basic mathematical evaluation capabilities for numerical
and symbolic computations. The LLM can use this to verify calculations by
writing out the specific expressions it wants to evaluate.

Capabilities:
- Evaluate numerical expressions
- Evaluate symbolic expressions
- Compare expressions for equivalence
- Simplify expressions
"""

from typing import Any, Dict, Optional, Tuple, Union
import sympy as sp


class MathCalculator:
    """
    A simple mathematical calculator for evaluating and verifying calculations.
    
    This class provides basic methods for:
    1. Evaluating numerical expressions
    2. Evaluating symbolic expressions
    3. Verifying calculation results
    4. Comparing expressions for equivalence
    
    The LLM should write out the specific calculations it wants to verify,
    and this calculator will evaluate them.
    """
    
    def __init__(self):
        """Initialize the calculator with common mathematical symbols."""
        # Mathematical constants
        self.constants = {
            'pi': sp.pi,
            'e': sp.E,
            'I': sp.I,
        }
    
    def evaluate_numerical(
        self,
        expression: str,
        variables: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Evaluate a mathematical expression numerically.
        
        Args:
            expression: Mathematical expression as string (e.g., "2*x + 3")
            variables: Dictionary of variable values (e.g., {"x": 5})
            
        Returns:
            Numerical result as float
            
        Examples:
            >>> calc.evaluate_numerical("2*5 + 3")
            13.0
            >>> calc.evaluate_numerical("2*x + 3", {"x": 5})
            13.0
            >>> calc.evaluate_numerical("sin(pi/2)")
            1.0
        """
        try:
            # Parse the expression
            expr = sp.sympify(expression, locals=self.constants)
            
            # Substitute variables if provided
            if variables:
                expr = expr.subs(variables)
            
            # Evaluate numerically
            result = float(expr.evalf())
            return result
                
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")
    
    def evaluate_symbolic(
        self,
        expression: str,
        variables: Optional[Dict[str, Any]] = None,
        simplify: bool = False
    ) -> str:
        """
        Evaluate a mathematical expression symbolically.
        
        Args:
            expression: Mathematical expression as string
            variables: Dictionary of variable values (can be symbolic or numeric)
            simplify: Whether to simplify the result
            
        Returns:
            Symbolic result as string
            
        Examples:
            >>> calc.evaluate_symbolic("x**2 + 2*x + 1")
            "x**2 + 2*x + 1"
            >>> calc.evaluate_symbolic("x**2 + 2*x + 1", simplify=True)
            "(x + 1)**2"
            >>> calc.evaluate_symbolic("(x+1)**2", simplify=True)
            "x**2 + 2*x + 1"
        """
        try:
            # Parse the expression
            expr = sp.sympify(expression, locals=self.constants)
            
            # Substitute variables if provided
            if variables:
                expr = expr.subs(variables)
            
            # Simplify if requested
            if simplify:
                expr = sp.simplify(expr)
            
            return str(expr)
                
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")
    
    def verify_calculation(
        self,
        expression: str,
        expected_result: Union[str, float],
        variables: Optional[Dict[str, float]] = None,
        tolerance: float = 1e-6
    ) -> Tuple[bool, str]:
        """
        Verify if a calculation is correct.
        
        Args:
            expression: Mathematical expression to evaluate
            expected_result: Expected result (can be numeric or symbolic)
            variables: Variable values for substitution
            tolerance: Tolerance for numeric comparison
            
        Returns:
            Tuple of (is_correct, explanation)
            
        Examples:
            >>> calc.verify_calculation("2 + 2", 4)
            (True, "✓ Calculation correct: 2 + 2 = 4.0")
            >>> calc.verify_calculation("x**2", "x*x")
            (True, "✓ Expressions are mathematically equivalent")
        """
        try:
            # Parse the expression
            expr = sp.sympify(expression, locals=self.constants)
            
            # If expected_result is numeric
            if isinstance(expected_result, (int, float)):
                # Substitute variables if provided
                if variables:
                    expr = expr.subs(variables)
                
                # Evaluate numerically
                result = float(expr.evalf())
                
                if abs(result - expected_result) <= tolerance:
                    return True, f"✓ Calculation correct: {expression} = {result}"
                else:
                    return False, f"✗ Calculation incorrect: {expression} = {result}, expected {expected_result}"
            
            # If expected_result is symbolic (string)
            else:
                expected_expr = sp.sympify(str(expected_result), locals=self.constants)
                
                # Substitute variables if provided
                if variables:
                    expr = expr.subs(variables)
                    expected_expr = expected_expr.subs(variables)
                
                # Check symbolic equivalence
                if sp.simplify(expr - expected_expr) == 0:
                    return True, f"✓ Expressions are mathematically equivalent"
                else:
                    return False, f"✗ Expressions are not equivalent: {expr} ≠ {expected_expr}"
                    
        except Exception as e:
            return False, f"✗ Error verifying calculation: {str(e)}"
    
    def compare_expressions(
        self,
        expr1: str,
        expr2: str,
        variables: Optional[Dict[str, float]] = None,
        tolerance: float = 1e-6
    ) -> Tuple[bool, str]:
        """
        Compare two mathematical expressions for equivalence.
        
        Args:
            expr1: First expression
            expr2: Second expression
            variables: Optional variable values for numeric comparison
            tolerance: Tolerance for numeric comparison
            
        Returns:
            Tuple of (are_equal, explanation)
            
        Examples:
            >>> calc.compare_expressions("x**2 + 2*x + 1", "(x+1)**2")
            (True, "✓ Expressions are symbolically equivalent")
            >>> calc.compare_expressions("2*x", "x + x")
            (True, "✓ Expressions are symbolically equivalent")
        """
        try:
            e1 = sp.sympify(expr1, locals=self.constants)
            e2 = sp.sympify(expr2, locals=self.constants)
            
            # Substitute variables if provided
            if variables:
                e1 = e1.subs(variables)
                e2 = e2.subs(variables)
            
            # Check if both are numbers after substitution
            if e1.is_number and e2.is_number:
                v1 = float(e1.evalf())
                v2 = float(e2.evalf())
                if abs(v1 - v2) <= tolerance:
                    return True, f"✓ Expressions are numerically equal: {v1}"
                else:
                    return False, f"✗ Expressions are not equal: {v1} ≠ {v2}"
            else:
                # Symbolic comparison
                if sp.simplify(e1 - e2) == 0:
                    return True, "✓ Expressions are symbolically equivalent"
                else:
                    return False, f"✗ Expressions are not equivalent: {e1} ≠ {e2}"
                    
        except Exception as e:
            return False, f"✗ Error comparing expressions: {str(e)}"
