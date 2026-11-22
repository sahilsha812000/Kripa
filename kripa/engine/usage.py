"""Normalize provider-reported usage data without deriving estimates."""

from math import isfinite


def record_provider_usage(owner, usage, input_names, output_names, cost_names=()):
    """Store only explicit nonnegative counts and provider-reported USD costs."""
    input_tokens = _first_valid_count(usage, input_names)
    output_tokens = _first_valid_count(usage, output_names)
    owner.last_usage = (
        {"input_tokens": input_tokens, "output_tokens": output_tokens}
        if input_tokens is not None or output_tokens is not None
        else None
    )
    owner.last_provider_cost_usd = _first_valid_cost(usage, cost_names)


def _value(source, name):
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _first_valid_count(usage, names):
    for name in names:
        value = _value(usage, name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _first_valid_cost(usage, names):
    for name in names:
        value = _value(usage, name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if value >= 0 and isfinite(value):
                return value
    return None
