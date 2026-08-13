"""Portfolio orchestrator (M3).

LangGraph graph that fans out to the agents, collects their signals, and asks
an LLM for an allocation honoring the user's amount, risk, and target, with
guardrails (weights sum to 100, respect risk band).
"""
