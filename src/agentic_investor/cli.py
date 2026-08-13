"""Console entry point (agentic-investor).

Prints resolved config so you can confirm the environment is wired up.
Real commands land in later milestones.
"""

from agentic_investor.config import get_settings


def main() -> None:
    s = get_settings()
    print("Agentic Investor scaffold OK\n")
    print(f"LLM model: {s.llm_model}")
    print(f"Orchestrator: {s.orchestrator_model}")
    print(f"Embedding model: {s.embedding_model}\n")

    keys = {
        "OpenAI": s.openai_api_key,
        "Anthropic": s.anthropic_api_key,
        "Gemini": s.gemini_api_key,
        "Finnhub": s.finnhub_api_key,
    }
    for name, value in keys.items():
        print(f"{name}: {'set' if value else 'not set (mock mode)'}")


if __name__ == "__main__":
    main()
