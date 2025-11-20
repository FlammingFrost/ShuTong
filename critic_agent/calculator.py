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
try:
    # Prefer LaTeX parsing when available
    from sympy.parsing.latex import parse_latex as _parse_latex
    _HAS_LATEX = True
except Exception:  # ImportError or other parser init issues
    _parse_latex = None
    _HAS_LATEX = False


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

    def _parse_expr(self, expression: Union[str, float, int]) -> sp.Expr:
        """
        Parse an input into a SymPy expression, preferring LaTeX.

        The method attempts LaTeX parsing first (using sympy.parsing.latex.parse_latex)
        and falls back to sympy.sympify with provided constants if LaTeX parsing is
        unavailable or fails.

        Args:
            expression: Input expression (LaTeX string, plain string, or numeric)

        Returns:
            SymPy expression
        """
        # Fast-path numerics
        if isinstance(expression, (int, float)):
            return sp.sympify(expression)

        expr_str = str(expression)

        # Heuristic: If expression contains LaTeX commands (backslash), try LaTeX first
        # Otherwise, try sympify first (for plain Python expressions)
        has_latex_commands = '\\' in expr_str or '^{' in expr_str
        
        if has_latex_commands and _HAS_LATEX and _parse_latex is not None:
            try:
                expr = _parse_latex(expr_str)
                # LaTeX parser creates symbols for constants like 'pi', 'e'
                # Substitute them with actual SymPy constants
                expr = expr.subs('pi', sp.pi).subs('e', sp.E).subs('I', sp.I)
                return expr
            except Exception:
                # Fall back to sympify if LaTeX parsing fails
                pass

        # Try SymPy string parsing with known constants
        try:
            # First, try parsing with transformations for implicit multiplication and power conversion
            from sympy.parsing.sympy_parser import (
                parse_expr, 
                standard_transformations, 
                implicit_multiplication_application,
                convert_xor
            )
            transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
            return parse_expr(expr_str, local_dict=self.constants, transformations=transformations)
        except Exception:
            # Fall back to basic sympify
            try:
                return sp.sympify(expr_str, locals=self.constants)
            except Exception:
                # If sympify fails and we haven't tried LaTeX yet, try it now
                if not has_latex_commands and _HAS_LATEX and _parse_latex is not None:
                    try:
                        expr = _parse_latex(expr_str)
                        # Substitute constants
                        expr = expr.subs('pi', sp.pi).subs('e', sp.E).subs('I', sp.I)
                        return expr
                    except Exception:
                        pass
                # Re-raise the original sympify error
                raise
    
    def evaluate_numerical(
        self,
        expression: str,
        variables: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Evaluate a mathematical expression numerically (LaTeX supported).
        
        Args:
            expression: Mathematical expression as LaTeX or plain string (e.g., r"2x+3", r"\sin(\pi/2)", "2*x+3")
            variables: Dictionary of variable values (e.g., {"x": 5})
            
        Returns:
            Numerical result as float
            
        Examples (LaTeX):
            >>> calc.evaluate_numerical(r"\tfrac{1}{2} + \tfrac{3}{4}")
            1.25
            >>> calc.evaluate_numerical(r"\sin\!\left(\tfrac{\pi}{6}\right) + \cos\!\left(\tfrac{\pi}{3}\right)")
            1.0
            >>> calc.evaluate_numerical(r"\int_{0}^{1} x^{2} \, dx")
            0.3333333333333333
            >>> calc.evaluate_numerical(r"2x + \tfrac{3}{2}", {"x": 5})
            11.5
        """
        try:
            # Parse (LaTeX first, then sympify)
            expr = self._parse_expr(expression)
            
            # Substitute variables if provided
            if variables:
                expr = expr.subs(variables)
            
            # Simplify/evaluate symbolic expressions before numeric conversion
            expr = sp.simplify(expr)
            
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
        Evaluate a mathematical expression symbolically (LaTeX supported).
        
        Args:
            expression: Mathematical expression as LaTeX or plain string
            variables: Dictionary of variable values (can be symbolic or numeric)
            simplify: Whether to simplify the result
            
        Returns:
            Symbolic result as string
            
        Examples (LaTeX):
            >>> calc.evaluate_symbolic(r"\frac{x^{2} - 1}{x - 1}")
            "(x**2 - 1)/(x - 1)"
            >>> calc.evaluate_symbolic(r"(x+1)^{2}", simplify=True)
            "x**2 + 2*x + 1"
            >>> calc.evaluate_symbolic(r"x^{2} + 2x + 1", simplify=True)
            "(x + 1)**2"
        """
        try:
            # Parse the expression (LaTeX first)
            expr = self._parse_expr(expression)
            
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
        Verify if a calculation is correct (supports LaTeX inputs).
        
        Args:
            expression: Mathematical expression to evaluate (LaTeX or plain)
            expected_result: Expected result (numeric or symbolic, LaTeX or plain)
            variables: Variable values for substitution
            tolerance: Tolerance for numeric comparison
            
        Returns:
            Tuple of (is_correct, explanation)
            
        Examples (LaTeX):
            >>> calc.verify_calculation(r"\tfrac{1}{2} + \tfrac{1}{3}", 5/6)
            (True, ...)
            >>> calc.verify_calculation(r"\int_{0}^{1} x^{2} \, dx", 1/3)
            (True, ...)
            >>> calc.verify_calculation(r"\frac{x^{2} - 1}{x - 1}", r"x + 1")
            (True, ...)
        """
        try:
            # Parse the expression (LaTeX first)
            expr = self._parse_expr(expression)
            
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
                expected_expr = self._parse_expr(str(expected_result))
                
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
        Compare two mathematical expressions for equivalence (LaTeX supported).
        
        Args:
            expr1: First expression (LaTeX or plain)
            expr2: Second expression (LaTeX or plain)
            variables: Optional variable values for numeric comparison
            tolerance: Tolerance for numeric comparison
            
        Returns:
            Tuple of (are_equal, explanation)
            
        Examples (LaTeX):
            >>> calc.compare_expressions(r"x^{2} + 2x + 1", r"(x+1)^{2}")
            (True, "✓ Expressions are symbolically equivalent")
            >>> calc.compare_expressions(r"\frac{x^{3} - 1}{x - 1}", r"x^{2} + x + 1")
            (True, ...)
        """
        try:
            e1 = self._parse_expr(expr1)
            e2 = self._parse_expr(expr2)
            
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
