"""
Example usage of the Solver Agent.
"""

from solver_agent import Solver


def main():
    """Run three example problems."""
    
    solver = Solver(model_name="gpt-4o", temperature=0.7)
    
    problems = [
        """Solve the following system of equations:
- $2x + 3y = 12$
- $x - y = 1$""",
        
        """Find the derivative of $f(x) = 3x^2 + 2x - 5$ at $x = 2$.""",
        
        """A rectangle has a perimeter of 24 cm and an area of 32 cm². Find its dimensions."""
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*80}")
        print(f"PROBLEM {i}:")
        print('='*80)
        print(problem)
        print(f"\n{'='*80}")
        print(f"SOLUTION {i}:")
        print('='*80)
        solution = solver.solve(problem)
        print(solution)
        print()


if __name__ == "__main__":
    main()
