from datetime import datetime, timezone

TOOL_ACCEPTING_PROVIDERS = {
    "groq",
    "openai",
    "openrouter",
    "gemini",
    "deepseek",
    # "ollama" # TODO
}

AGENTS_REGISTRY = {}

date_time_func = {
    "format_date": lambda data=None: datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
}
