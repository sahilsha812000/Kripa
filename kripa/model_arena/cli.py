"""Command-line entry point for the Model Arena."""

from kripa.config import get_settings
from kripa.model_arena.report import format_report
from kripa.model_arena.runner import ModelArena, PROVIDER_NAMES


def run_arena(arguments=None, settings=None, arena=None, store=None):
    names = list(arguments or [])
    invalid = sorted(set(names).difference(PROVIDER_NAMES))
    if invalid:
        print("Unknown provider(s): {}. Choose from: {}.".format(
            ", ".join(invalid), ", ".join(PROVIDER_NAMES),
        ))
        return 2
    try:
        report = (arena or ModelArena(names or None)).run(settings or get_settings())
    except ValueError as error:
        print(str(error))
        return 2
    print(format_report(report))
    try:
        if store is None:
            from kripa.model_router.store import ModelRouterStore
            store = ModelRouterStore()
        count = store.save_arena_report(report)
        print("\nSaved {} task/model benchmark summaries to .kripa/model_router.json.".format(count))
    except (OSError, TypeError, ValueError) as error:
        print("\nError: benchmark results could not be saved ({}).".format(type(error).__name__))
        return 1
    return 0
