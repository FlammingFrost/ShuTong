A agent built with LangGraph will be used for following purposes:


Functionality 1: Give step-by-step solution to math problems.
- Input:
  - Math problem in markedown format.
- Output:
  - Step-by-step solution in markedown format. With following formats:
    - Use ```### Step [N]: [Description]``` to indicate each step.
    - Use LaTeX format for all mathematical expressions. Use `$$ ... $$` for block equations and `$ ... $` for inline equations.

Functionality 2: Refined the math solution with feedback from reflector agent.
- Input:
  - Previous solution in markedown format.
  - Feedback from critic agent in markedown format.
- Output:
  - Refined step-by-step solution in markedown format. With following formats:
    - Use ```### Step [N]: [Description]``` to indicate each step.
    - Use LaTeX format for all mathematical expressions. Use `$$ ... $$` for block equations and `$ ... $` for inline equations.