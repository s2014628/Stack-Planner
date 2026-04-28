"""
Direct Generate Pipeline

Faster alternative to the full StackPlanner agent pipeline.
Instead of running a multi-step agent, directly calls an LLM to:
  1. Solve the problem (get reasoning + prediction)
  2. Generate structured experience (problem_type, practice, lessons_learned)

Output format is fully compatible with qa_bench_pipeline's per_run_experiences.json.
"""
