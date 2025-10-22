The folder builds the agent system components, which include:
- Solver agents
  - Provide step-by-step solutions to math problems.
  - Revise solutions based on user feedback.
- Critic agents
  - Evaluate the quality of solutions provided by solver agents.
  - Provide constructive feedback for improvement.
  - Find the knowledge points that used in each step of the solution.
- Knowledge retriever (Not implemented yet)
  - Review the collection of knowledge points identified by critic agents, combine similar points, and output a unified knowledge base for future reference.

## Agent pipelines
[Math Problem] → Solver Agent → [Initial Solution]
[Initial Solution] → [Steps of Solution] (with Step extractor)
    For each Step:
    [Step] → Critic Agent → [Feedback] + [Knowledge Points]
  Add [Feedback] to [Collections of Feedbacks]
  Add [Knowledge Points] to [Collections of Knowledge Points]
([Initial Solution] + [Collections of Feedbacks]) → Solver Agent → [Refined Solution]
[Collections of Knowledge Points] → Knowledge Retriever → [Unified Knowledge Base]
