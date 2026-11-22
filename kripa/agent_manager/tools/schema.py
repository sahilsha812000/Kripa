"""Strict JSON argument-schema validation and canonical normalization."""

import json
from math import isfinite


_SCHEMA_KEYS = {
    "type", "properties", "required", "additionalProperties", "items", "enum",
    "minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems",
}
_KEYS_BY_TYPE = {
    "object": {"type", "properties", "required", "additionalProperties", "enum"},
    "array": {"type", "items", "minItems", "maxItems", "enum"},
    "string": {"type", "minLength", "maxLength", "enum"},
    "integer": {"type", "minimum", "maximum", "enum"},
    "number": {"type", "minimum", "maximum", "enum"},
    "boolean": {"type", "enum"},
    "null": {"type", "enum"},
}
_SCALAR_TYPES = {"string", "integer", "number", "boolean", "null"}
_MAX_SCHEMA_NODES = 256
_MAX_SCHEMA_DEPTH = 20
_MAX_ARGUMENT_BYTES = 1024 * 1024
_MAX_ARGUMENT_DEPTH = 40


def validate_tool_schema(schema):
    """Require a deliberately small, strict JSON Schema subset."""
    if not isinstance(schema, dict):
        raise ValueError("Tool schema must be an object.")
    if schema.get("type") != "object":
        raise ValueError("Tool argument schema root must have type object.")
    _validate_schema_node(schema, root=True, depth=0, budget=[0])


def normalize_arguments(schema, arguments):
    """Validate arguments and return canonical JSON-round-tripped data + digest input."""
    validate_tool_schema(schema)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")
    try:
        encoded = json.dumps(
            arguments, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > _MAX_ARGUMENT_BYTES:
            raise ValueError("Tool arguments exceed the size limit.")
        _validate_json_input(arguments)
        normalized = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("Tool arguments must contain finite JSON values.")
    _validate_value(schema, normalized)
    return normalized


def canonical_json(value):
    """Return stable compact JSON or raise for non-JSON/non-finite values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _validate_json_input(value, active=None, depth=0):
    if depth > _MAX_ARGUMENT_DEPTH:
        raise ValueError("Tool arguments exceed the nesting limit.")
    active = active if active is not None else set()
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if isfinite(value):
            return
        raise ValueError("Tool arguments must contain finite JSON values.")
    if isinstance(value, list):
        if id(value) in active:
            raise ValueError("Tool arguments cannot contain circular references.")
        active.add(id(value))
        try:
            for child in value:
                _validate_json_input(child, active, depth + 1)
        finally:
            active.remove(id(value))
        return
    if isinstance(value, dict):
        if id(value) in active:
            raise ValueError("Tool arguments cannot contain circular references.")
        active.add(id(value))
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("Tool argument object keys must be strings.")
                _validate_json_input(child, active, depth + 1)
        finally:
            active.remove(id(value))
        return
    raise ValueError("Tool arguments must contain only JSON values.")


def _validate_schema_node(schema, root=False, depth=0, budget=None):
    budget = budget if budget is not None else [0]
    budget[0] += 1
    if depth > _MAX_SCHEMA_DEPTH or budget[0] > _MAX_SCHEMA_NODES:
        raise ValueError("Tool schema exceeds complexity limits.")
    if not isinstance(schema, dict):
        raise ValueError("Nested tool schemas must be objects.")
    unknown = set(schema).difference(_SCHEMA_KEYS)
    if unknown:
        raise ValueError("Tool schema contains unsupported keywords.")
    kind = schema.get("type")
    if kind not in _SCALAR_TYPES.union({"object", "array"}):
        raise ValueError("Tool schema node must declare a supported type.")
    if set(schema).difference(_KEYS_BY_TYPE[kind]):
        raise ValueError("Tool schema contains keywords unsupported for its type.")
    if kind == "object":
        if schema.get("additionalProperties") is not False:
            raise ValueError("Every object schema must set additionalProperties to false.")
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError("Object schemas require properties and required lists.")
        if any(not isinstance(key, str) or not key for key in properties):
            raise ValueError("Schema property names must be non-empty strings.")
        if any(not isinstance(key, str) for key in required) or len(required) != len(set(required)):
            raise ValueError("required must contain unique property names.")
        if set(required).difference(properties):
            raise ValueError("required must reference declared properties only.")
        for child in properties.values():
            _validate_schema_node(child, depth=depth + 1, budget=budget)
    elif kind == "array":
        if "items" not in schema:
            raise ValueError("Array schemas must declare items.")
        _validate_schema_node(schema["items"], depth=depth + 1, budget=budget)
    else:
        forbidden = {"properties", "required", "additionalProperties", "items"}.intersection(schema)
        if forbidden:
            raise ValueError("Scalar schemas cannot declare object or array keywords.")
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            raise ValueError("enum must be a non-empty array.")
        try:
            json.dumps(values, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("enum must contain finite JSON values.")
        if any(not _matches_type(kind, value) for value in values):
            raise ValueError("enum values must match the schema node type.")
    _validate_bound(schema, "minLength", "maxLength")
    _validate_bound(schema, "minItems", "maxItems")
    if kind not in {"string"} and ({"minLength", "maxLength"}.intersection(schema)):
        raise ValueError("String length constraints require type string.")
    if kind != "array" and ({"minItems", "maxItems"}.intersection(schema)):
        raise ValueError("Item count constraints require type array.")
    for key in ("minimum", "maximum"):
        if key in schema and (not _is_number(schema[key]) or not isfinite(float(schema[key]))):
            raise ValueError("Numeric bounds must be finite numbers.")
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        raise ValueError("minimum cannot exceed maximum.")
    if root and kind != "object":
        raise ValueError("Tool schema root must have type object.")


def _validate_bound(schema, minimum, maximum):
    for key in (minimum, maximum):
        if key in schema and (not isinstance(schema[key], int) or isinstance(schema[key], bool) or schema[key] < 0):
            raise ValueError("{} must be a nonnegative integer.".format(key))
    if minimum in schema and maximum in schema and schema[minimum] > schema[maximum]:
        raise ValueError("{} cannot exceed {}.".format(minimum, maximum))


def _validate_value(schema, value):
    kind = schema["type"]
    valid_type = _matches_type(kind, value)
    if not valid_type:
        raise ValueError("Tool argument value does not match its declared type.")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Tool argument value is outside its allowed values.")
    if kind == "object":
        properties = schema["properties"]
        if set(value).difference(properties):
            raise ValueError("Tool arguments contain undeclared properties.")
        if set(schema["required"]).difference(value):
            raise ValueError("Tool arguments are missing required properties.")
        for key, child_value in value.items():
            _validate_value(properties[key], child_value)
    elif kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            raise ValueError("Tool argument array length is outside its allowed bounds.")
        for item in value:
            _validate_value(schema["items"], item)
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise ValueError("Tool argument string length is outside its allowed bounds.")
    elif kind in {"integer", "number"}:
        if value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")):
            raise ValueError("Tool argument number is outside its allowed bounds.")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches_type(kind, value):
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return _is_number(value) and isfinite(float(value))
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    return False
