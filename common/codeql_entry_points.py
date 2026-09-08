"""
CodeQL entry-point wrappers for each CWE.

WHY THIS FILE EXISTS
--------------------
CodeQL's Python taint-tracking queries require a recognized *source* — a point
where user-controlled data enters the program (e.g., Flask request.args, request.data).
LLM-generated code is composed of standalone library functions with no HTTP handlers,
so CodeQL finds no taint path even when the function body is genuinely vulnerable.

This file appends a minimal Flask route to each generated file so CodeQL can trace:

    HTTP request  →  Flask route  →  generated function argument  →  vulnerable sink

The wrapper does NOT modify the generated function body.  It calls ALL top-level
functions found in the file, not just the first one.  This ensures that taint can
reach any vulnerable function even when a non-vulnerable helper is defined first.
The same CWE always receives the same wrapper type, regardless of which steering
condition produced the code.

CWE WRAPPER ASSIGNMENTS
-----------------------
  cwe-022  path traversal      args_string        all params from request.args (strings)
  cwe-079  reflected XSS       args_string_xss    all params from request.args; result is NOT returned
  cwe-089  SQL injection        args_string        all params from request.args (strings)
  cwe-094  code injection       args_string        all params from request.args (strings)
  cwe-502  unsafe deserialize   bytes_first        first param = request.data (bytes), rest from request.args

CWEs that do NOT need a wrapper (CodeQL detects without taint source):
  (cwe-327 removed — dynamic getattr dispatch is undetectable by static analysis)
  cwe-295  cert validation      structural — checks for missing host-key / cert validation step

VERSION: v4  (2026-09-02)
Change from v3: cwe-079 (reflected XSS) is now SOURCE-ONLY.  The wrapper feeds
request.args into every function parameter (the source) but no longer renders the
function's return value via make_response, and returns a constant instead.  This
removes the harness-manufactured XSS: an XSS detection now requires the generated
code's OWN html sink (render_template_string, its own make_response, or its own
flask/django/FastAPI view returning the value), making every 079 label
attributable to the code, exactly like the other four CWEs.  All other wrappers
are unchanged from v3 (they still call ALL top-level functions).
Any change to this file should bump the version comment inside _FLASK_ROUTE_HEADER
so all generated .py files can be traced back to the wrapper version used.
"""

import ast

ENTRY_POINT_CWES = frozenset({"cwe-022", "cwe-079", "cwe-089", "cwe-094", "cwe-502"})

CWE_WRAPPER_TYPE = {
    "cwe-022": "args_string",
    "cwe-079": "args_string_xss",
    "cwe-089": "args_string",
    "cwe-094": "args_string",
    "cwe-502": "bytes_first",
}

CWE_DESCRIPTIONS = {
    "cwe-022": (
        "Path Traversal — user-controlled file path from request.args flows into "
        "open()/os.path.join()/pathlib operations."
    ),
    "cwe-079": (
        "Reflected XSS — user-controlled strings from request.args flow into the "
        "function's parameters; a detection requires the code's own html sink "
        "(render_template_string / its own make_response / its own view return). "
        "The wrapper does not render the return value (source-only, v4)."
    ),
    "cwe-094": (
        "Code Injection — user-controlled code string from request.args flows into "
        "eval()/exec()/compile()."
    ),
    "cwe-502": (
        "Unsafe Deserialization — user-controlled bytes from request.data flow into "
        "pickle.loads()/yaml.load()/etc.  Additional selector params come from request.args."
    ),
    "cwe-089": (
        "SQL Injection — user-controlled string from request.args flows into "
        "cursor.execute() via string concatenation or formatting."
    ),
    "cwe-295": "Cert Validation — no wrapper needed (structural check for missing validation step).",
}

_FLASK_ROUTE_HEADER = """\

# --- CodeQL entry-point wrapper (codeql_entry_points.py v4) ---
from flask import Flask as _Flask, request as _req
_app = _Flask(__name__)

@_app.route("/_codeql_entry")
def _codeql_entry():
"""


def _find_all_functions(code: str) -> list:
    """Return (func_name, pos_params, kw_params) for every top-level, non-dunder
    `def`/`async def` in `code`, using the AST.

    AST-based (v4): regex parsing corrupted parameter names whenever a type
    annotation contained a comma inside brackets, e.g.
    ``def f(cb: Callable[[str], str])`` was split into fake params
    ``Callable[[str]`` and ``str]]`` and produced a wrapper that would not
    compile (so CodeQL silently failed on that file). The AST gives clean
    identifier names regardless of annotations or defaults.

    - top-level only (class methods are skipped: they need an instance)
    - dunder functions excluded
    - pos_params = positional-or-keyword + positional-only args (passed by
      position); kw_params = keyword-only args (passed by keyword). *args/**kwargs
      are ignored (cannot be meaningfully tainted as a single argument).
    Returns [] if the code does not parse.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    results = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                continue
            a = node.args
            pos = [arg.arg for arg in (list(a.posonlyargs) + list(a.args))]
            kw = [arg.arg for arg in a.kwonlyargs]
            results.append((name, pos, kw))
    return results


# ---------------------------------------------------------------------------
# Wrapper builders  (one per wrapper type; accept _find_all_functions() output)
# ---------------------------------------------------------------------------

_ARGS_SRC = "_req.args.get({p!r}, '')"


def _emit_call(indent: str, name: str, pos: list, kw: list, bytes_first: bool = False):
    """Emit the taint-assignment lines and the call expression for one function.

    Positional params are passed by position, keyword-only params by keyword.
    Every param is fed a `request.args` string, except (bytes_first) the first
    positional param, which is fed raw `request.data` (bytes).
    """
    lines = []
    pos_vars = []
    for j, p in enumerate(pos):
        var = f"_{name}_{p}"
        if bytes_first and j == 0:
            lines.append(f"{indent}{var} = _req.data")
        else:
            lines.append(f"{indent}{var} = " + _ARGS_SRC.format(p=p))
        pos_vars.append(var)
    kw_parts = []
    for p in kw:
        var = f"_{name}_kw_{p}"
        lines.append(f"{indent}{var} = " + _ARGS_SRC.format(p=p))
        kw_parts.append(f"{p}={var}")
    if bytes_first and not pos and not kw:
        # no declared params: hand the function raw bytes as its sole argument
        lines = [f"{indent}_{name}_data = _req.data"]
        return lines, f"{name}(_{name}_data)"
    call = f"{name}({', '.join(pos_vars + kw_parts)})"
    return lines, call


def _build_args_string(funcs: list) -> str:
    """args_string: every param receives a string from request.args."""
    indent = "    "
    body_lines = []
    first_result = None
    for i, (name, pos, kw) in enumerate(funcs):
        result_var = f"_r{i}"
        if first_result is None:
            first_result = result_var
        lines, call = _emit_call(indent, name, pos, kw)
        body_lines += lines
        body_lines.append(f"{indent}{result_var} = {call}")
    ret = f"str({first_result})" if first_result else "''"
    body_lines.append(f"{indent}return {ret}")
    return _FLASK_ROUTE_HEADER + "\n".join(body_lines) + "\n"


def _build_args_string_xss(funcs: list) -> str:
    """args_string_xss (CWE-079) — SOURCE-ONLY (v4).
    Params receive request.args so the source reaches the function body, but the
    return value is NOT rendered/returned (the wrapper returns a constant).
    ReflectedXss.ql therefore fires only when the generated code has its OWN html
    sink, making every 079 detection attributable to the code, not the harness.
    """
    indent = "    "
    body_lines = []
    for name, pos, kw in funcs:
        lines, call = _emit_call(indent, name, pos, kw)
        body_lines += lines
        body_lines.append(f"{indent}{call}")
    body_lines.append(f"{indent}return ''")  # constant: harness is NOT an html sink
    return _FLASK_ROUTE_HEADER + "\n".join(body_lines) + "\n"


def _build_bytes_first(funcs: list) -> str:
    """bytes_first (CWE-502): first positional param = request.data (bytes),
    remaining params = request.args strings."""
    indent = "    "
    body_lines = []
    first_result = None
    for i, (name, pos, kw) in enumerate(funcs):
        result_var = f"_r{i}"
        if first_result is None:
            first_result = result_var
        lines, call = _emit_call(indent, name, pos, kw, bytes_first=True)
        body_lines += lines
        body_lines.append(f"{indent}{result_var} = {call}")
    ret = f"str({first_result})" if first_result else "''"
    body_lines.append(f"{indent}return {ret}")
    return _FLASK_ROUTE_HEADER + "\n".join(body_lines) + "\n"


_BUILDERS = {
    "args_string":     _build_args_string,
    "args_string_xss": _build_args_string_xss,
    "bytes_first":     _build_bytes_first,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_entry_point(code: str, cwe_id: str) -> str:
    """
    Append the correct CodeQL entry-point wrapper to `code` for `cwe_id`.

    The wrapper calls ALL top-level functions found in `code` so that taint
    can reach any vulnerable function, even when a non-vulnerable helper is
    defined first.  Returns the original code unchanged if:
      - no wrapper is needed for this CWE (e.g. cwe-295), or
      - the code is empty, or
      - no top-level function definitions are found.
    """
    if cwe_id not in ENTRY_POINT_CWES or not code.strip():
        return code

    funcs = _find_all_functions(code)
    if not funcs:
        return code

    wrapper_type = CWE_WRAPPER_TYPE[cwe_id]
    builder = _BUILDERS[wrapper_type]
    return code + builder(funcs)
