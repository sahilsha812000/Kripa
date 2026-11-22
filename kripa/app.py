"""Interactive command-line application."""

from kripa.config import get_settings
from kripa.engine import Conversation, create_provider
from kripa.engine.errors import ProviderError


def main(argv=None) -> int:
    """Run chat or the read-only model report command."""
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "agents":
        from kripa.agent_manager.cli import run_agents

        return run_agents(arguments[1:])
    if arguments == ["models", "report"]:
        from kripa.model_updates.cli import run_report

        return run_report()
    if arguments and arguments[0] == "router":
        from kripa.model_router.cli import run_status

        if len(arguments) < 2 or arguments[1] != "status":
            print("Usage: python -m kripa router status [--refresh]")
            return 2
        return run_status(arguments[2:])
    if arguments and arguments[0] == "arena":
        from kripa.model_arena.cli import run_arena

        return run_arena(arguments[1:])
    if arguments:
        print("Usage: python -m kripa [agents <list|create|show|update|retire|run> ... | models report | router status [--refresh] | arena [gemini groq openrouter ollama ...]]")
        return 2
    try:
        settings = get_settings()
        provider = create_provider(settings)
    except ProviderError as error:
        print(f"Configuration error: {error}")
        return 1

    from kripa.model_router.shadow import build_shadow_router

    conversation = Conversation(provider, shadow_router=build_shadow_router(settings))
    print(f"Kripa is ready ({settings.provider}). Type /quit to exit.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if text.lower() in {"/quit", "/exit"}:
            print("Goodbye.")
            return 0
        if not text:
            continue
        try:
            print(f"Kripa: {conversation.send(text)}")
        except ProviderError as error:
            print(f"Kripa: {error}")
    return 0
