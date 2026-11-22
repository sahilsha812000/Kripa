"""Correctness checks for fixed benchmark tasks.

Generated Python is parsed and interpreted by a deliberately tiny, allowlisted
interpreter. It is never passed to exec/eval, a shell, or a subprocess.
"""

import ast
import re


def check_response(task_id, response, expected):
    """Return True/False for machine-checkable tasks, or None for manual review."""
    if task_id == "reasoning":
        return _has_expected_final_answer(response, expected)
    if task_id == "instruction_following":
        return response.strip() == expected
    if task_id == "coding":
        return _check_python_cases(response, expected)
    if task_id == "writing":
        return None
    if task_id == "general_conversation":
        return _check_practical_suggestion(response, expected)
    if task_id == "research":
        return _check_source_grounded_answer(response, expected)
    if task_id == "summarization":
        return _check_key_fact_summary(response, expected)
    raise ValueError("Unknown arena task: {}".format(task_id))


def _sentence_count(response):
    return len(re.findall(r"[.!?](?=\s|$)", response.strip()))


def _contains_required_groups(response, groups):
    normalized = response.casefold()
    return all(any(option.casefold() in normalized for option in group) for group in groups)


def _check_practical_suggestion(response, expected):
    if not isinstance(expected, dict):
        return False
    return (
        _sentence_count(response) == expected.get("sentences")
        and any(term.casefold() in response.casefold() for term in expected.get("required_any", ()))
    )


def _check_source_grounded_answer(response, expected):
    if not isinstance(expected, dict):
        return False
    citation = expected.get("citation", "")
    return (
        _sentence_count(response) == expected.get("sentences")
        and bool(citation)
        and citation.casefold() in response.casefold()
        and _contains_required_groups(response, expected.get("required_groups", ()))
    )


def _check_key_fact_summary(response, expected):
    if not isinstance(expected, dict):
        return False
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    bullet_prefixes = ("- ", "* ", "• ")
    return (
        len(lines) == expected.get("bullet_count")
        and all(line.startswith(bullet_prefixes) for line in lines)
        and _contains_required_groups(response, expected.get("required_groups", ()))
    )


def _has_expected_final_answer(response, expected):
    """Check the final standalone numeric answer, allowing simple explanation."""
    normalized = response.strip().replace(",", "")
    matches = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)", normalized)
    if not matches:
        return False
    try:
        return int(matches[-1]) == int(expected)
    except (TypeError, ValueError):
        return False


def _check_python_cases(response, cases):
    source = _extract_code(response)
    if not source or len(source) > 10000:
        return False
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    nodes = list(ast.walk(tree))
    if len(nodes) > 250:
        return False
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or any(
        not isinstance(node, (ast.FunctionDef, ast.Expr)) for node in tree.body
    ):
        return False
    function = functions[0]
    if (
        function.name != "is_palindrome"
        or function.decorator_list
        or function.args.vararg
        or function.args.kwarg
        or function.args.kwonlyargs
        or function.args.defaults
        or function.args.kw_defaults
        or len(function.args.args) != 1
        or function.args.args[0].arg != "text"
    ):
        return False

    try:
        for argument, expected in cases:
            result = _run_simple_function(function, argument)
            if not isinstance(result, bool) or result is not expected:
                return False
    except Exception:
        return False
    return True


def _extract_code(response):
    fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", response, re.IGNORECASE | re.DOTALL)
    return (fenced.group(1) if fenced else response).strip()


def _run_simple_function(function, text):
    """Interpret only assignments, returns, literals, and selected string ops."""
    environment = {function.args.args[0].arg: text}
    returned = False
    result = None
    for statement in function.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            if isinstance(statement.value.value, str):  # optional docstring
                continue
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise ValueError("Only simple local assignments are supported.")
            name = statement.targets[0].id
            if name == function.args.args[0].arg:
                raise ValueError("The input argument cannot be reassigned.")
            environment[name] = _evaluate(statement.value, environment)
        elif isinstance(statement, ast.Return):
            result = _evaluate(statement.value, environment) if statement.value else None
            returned = True
            break
        else:
            raise ValueError("Unsupported function statement.")
    if not returned:
        raise ValueError("Function has no return statement.")
    return result


def _evaluate(node, environment):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, bool)):
        return node.value
    if isinstance(node, ast.Name) and node.id in environment:
        return environment[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = _evaluate(node.func.value, environment)
        args = [_evaluate(argument, environment) for argument in node.args]
        if node.keywords:
            raise ValueError("Keyword arguments are unsupported.")
        method = node.func.attr
        if isinstance(base, str):
            if method in ("lower", "casefold", "split", "isalnum") and not args:
                return getattr(base, method)()
            if method == "replace" and len(args) == 2 and all(isinstance(arg, str) for arg in args):
                return base.replace(args[0], args[1])
            if method == "join" and len(args) == 1 and isinstance(args[0], list):
                if all(isinstance(item, str) for item in args[0]):
                    return base.join(args[0])
        raise ValueError("Unsupported method call.")
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        base = _evaluate(node.value, environment)
        slice_node = node.slice
        if (
            isinstance(base, str)
            and slice_node.lower is None
            and slice_node.upper is None
            and isinstance(slice_node.step, ast.UnaryOp)
            and isinstance(slice_node.step.op, ast.USub)
            and isinstance(slice_node.step.operand, ast.Constant)
            and slice_node.step.operand.value == 1
        ):
            return base[::-1]
        raise ValueError("Only reverse-string slicing is supported.")
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left = _evaluate(node.left, environment)
        right = _evaluate(node.comparators[0], environment)
        if isinstance(node.ops[0], ast.Eq):
            return left == right
        if isinstance(node.ops[0], ast.NotEq):
            return left != right
    raise ValueError("Unsupported expression.")
