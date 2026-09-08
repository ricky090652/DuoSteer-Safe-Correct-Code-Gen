"""
extract_code.py

Stage 1, step 4: extract the code from raw model generations (res_*.jsonl with a `predicted_code`
list of N samples per record) as COMPLETE standalone Python snippets.

For every sample:
  - complete as-is                          -> keep
  - a bare completion of a `## COMPLETE CODE HERE` template task (the template
    is the task's `question`)
                                            -> splice the completion back into the
                                               template (correct indent) and keep
                                               if the result is complete
  - still not complete after that           -> drop (nothing is fabricated)

"Complete" means: compiles as a module, has no top-level `return`/`yield` (a
bare-body fragment), and, when it defines no function or class, references no
undefined *local* name (a name that is not imported, not a builtin, and not a
known stdlib/framework symbol). Code with a missing import (uses Flask without
importing it) or app-level globals (`app`, `db`) is NOT dropped: the CodeQL
wrapper supplies taint through parameters, so such code is still analyzable.
Only genuinely torn fragments are dropped.

Output: one file per input under --out_dir with the same schema, `predicted_code`
replaced by the kept and spliced snippets (raw code, no fences) and a new
`predicted_code_meta` list holding the per-sample status. Records whose samples
all drop are still written (empty `predicted_code`) so counts stay visible.

Usage:
  python dataset_construction/extract_code.py \
      --in_glob 'data/code_gen_results_sampling/res_*.jsonl' \
      --out_dir data/code_gen_extracted/llama --dry_run
  (drop --dry_run to write; add --examples N to print spliced samples)
"""
from __future__ import annotations

import argparse
import ast
import builtins
import glob
import json
import os
import re
import textwrap
import warnings
from collections import Counter

MARKER = "## COMPLETE CODE HERE"
FENCE = re.compile(r"```(?:python)?\s*\n?(.*?)\n?```", re.DOTALL)
BRACKET_LINE = re.compile(r"^\s*\[[^\]]*\]\s*$")
CODE_START = re.compile(
    r"^(import |from |def |class |@|async |if |for |while |with |try|except|"
    r"finally|else|elif |return|raise |yield|global |nonlocal |del |assert |"
    r"print\(|[A-Za-z_][\w.\[\]]*\s*(=|\+=|-=|\*=|/=|\()|#!)")

BUILTINS = set(dir(builtins)) | {
    "__name__", "__file__", "__doc__", "self", "cls", "__all__", "__spec__",
    "__main__", "__builtins__", "__package__", "__loader__",
}
# stdlib / common-framework names that, if unimported, mean "missing import"
# (a real but harmless defect) rather than a torn fragment.
IMPORTABLE = set("""
os sys io re json time datetime shutil tempfile zipfile tarfile subprocess glob
hashlib base64 pickle marshal yaml requests random math logging pathlib socket ssl
secrets string collections functools itertools sqlite3 csv uuid threading queue
typing struct binascii zlib gzip http urllib xml html email smtplib ftplib argparse
Flask request jsonify redirect url_for render_template render_template_string abort
make_response send_file send_from_directory session flash Response current_app Blueprint
secure_filename Markup escape Path BytesIO StringIO Template FastAPI APIRouter Query
Body Request Depends HTTPException app db np pd numpy pandas
""".split())


def group_from_filename(path: str) -> str:
    """Prompt group of a generation file, from its name.

    'safe' or 'benign' in the name -> safe (benign prompt);
    'vuln_generic'                 -> vuln_generic (generic vulnerability-eliciting prompt);
    'vuln'                         -> vuln (CWE-specific vulnerability-eliciting prompt).
    """
    b = os.path.basename(path).lower()
    if "vuln_generic" in b:
        return "vuln_generic"
    if "safe" in b or "benign" in b:
        return "safe"
    if "vuln" in b:
        return "vuln"
    raise ValueError(f"cannot infer the prompt group from file name: {b} "
                     "(expected 'safe'/'benign', 'vuln', or 'vuln_generic' in the name)")


def strip_fences(text: str) -> str:
    m = FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


def is_code_line(line: str) -> bool:
    if not line.strip():
        return False
    if line[:1] in " \t":
        return True
    if MARKER in line:
        return True
    s = line.strip()
    if s in (")", "]", "}"):
        return True
    return bool(CODE_START.match(line))


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _bound_names(tree: ast.AST) -> set:
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                    names.add(arg.arg)
                if a.vararg:
                    names.add(a.vararg.arg)
                if a.kwarg:
                    names.add(a.kwarg.arg)
        elif isinstance(n, ast.Import):
            for al in n.names:
                names.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for al in n.names:
                if al.name != "*":
                    names.add(al.asname or al.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
    return names


def _undef_local_names(tree: ast.AST) -> set:
    bound = _bound_names(tree)
    undef = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in bound and n.id not in BUILTINS and n.id not in IMPORTABLE:
                undef.add(n.id)
    return undef


def _compiles(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(code, "<snippet>", "exec")
        return True
    except SyntaxError:
        return False
    except (ValueError, TypeError):   # e.g. null bytes
        return False


def is_complete(code: str) -> bool:
    """Complete standalone snippet: compiles as a module, and if it defines no
    function or class it must not reference undefined local names.

    compile() (not just ast.parse) is the syntactic gate: compile() rejects
    `return`/`yield` outside a function anywhere in the module, the bare-body
    fragment case, which ast.parse silently accepts."""
    if not code.strip():
        return False
    if not _compiles(code):
        return False
    tree = ast.parse(code)               # safe: already compiled
    has_def = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  for n in tree.body)
    if not has_def and _undef_local_names(tree):
        return False
    return True


def extract_template_lines(template_text: str):
    lines = [l for l in template_text.splitlines() if not BRACKET_LINE.match(l)]
    mark_idx = next((i for i, l in enumerate(lines) if MARKER in l), None)
    if mark_idx is None:
        return None
    start = mark_idx
    for i in range(mark_idx - 1, -1, -1):
        if is_code_line(lines[i]) or not lines[i].strip():
            start = i
        else:
            break
    while start < mark_idx and not lines[start].strip():
        start += 1
    end = mark_idx
    for i in range(mark_idx + 1, len(lines)):
        if is_code_line(lines[i]) or not lines[i].strip():
            end = i
        else:
            break
    while end > mark_idx and not lines[end].strip():
        end -= 1
    seg = lines[start:end + 1]
    return seg or None


def splice_into_template(template_text: str, completion: str):
    """Splice a bare completion into the task's code template at the marker."""
    tmpl = extract_template_lines(template_text)
    if not tmpl:
        return None
    mi = next((i for i, l in enumerate(tmpl) if MARKER in l), None)
    if mi is None:
        return None
    target = None
    for i in range(mi + 1, len(tmpl)):
        if tmpl[i].strip():
            target = indent_of(tmpl[i]); break
    if target is None:
        for i in range(mi - 1, -1, -1):
            if tmpl[i].strip():
                target = indent_of(tmpl[i]); break
    if target is None:
        target = indent_of(tmpl[mi])
    body = textwrap.dedent(completion).strip("\n")
    reindented = "\n".join((" " * target + ln) if ln.strip() else ln
                           for ln in body.splitlines())
    out_lines = tmpl[:mi] + [reindented] + tmpl[mi + 1:]
    code = "\n".join(out_lines).rstrip() + "\n"
    if not _compiles(code):
        return None
    return code


def process_sample(raw_sample: str, template_text):
    """Return (status, code_or_None). status in {complete, spliced, dropped}."""
    code = strip_fences(raw_sample)
    if is_complete(code):
        return "complete", code
    if template_text and MARKER in template_text:
        fixed = splice_into_template(template_text, code)
        if fixed is not None and is_complete(fixed):
            return "spliced", fixed.rstrip("\n")
    return "dropped", None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--in_glob", required=True, help="raw generation files (res_*.jsonl)")
    ap.add_argument("--out_dir", required=True, help="where the extracted files go (same file names)")
    ap.add_argument("--dry_run", action="store_true", help="report counts without writing")
    ap.add_argument("--examples", type=int, default=0, help="print N spliced samples (completion merged into its template)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.in_glob))
    if not paths:
        raise SystemExit(f"no files match {args.in_glob}")
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    stat = Counter()
    by_src = Counter()
    shown = 0
    for path in paths:
        rows_out = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tmpl = r.get("question") or (r["messages"][0]["content"] if r.get("messages") else None)
            preds = r.get("predicted_code", []) or []
            if isinstance(preds, str):
                preds = [preds]
            kept, meta = [], []
            for s in preds:
                status, code = process_sample(s, tmpl)
                stat[status] += 1
                by_src[(r.get("source", "?"), status)] += 1
                meta.append(status)
                if code is not None:
                    kept.append(code)
                    if status == "spliced" and shown < args.examples:
                        shown += 1
                        print(f"\n--- SPLICED {r['id']} (src={r.get('source')}) ---")
                        print("[raw completion]"); print(strip_fences(s)[:200])
                        print("[spliced]"); print(code[:400])
            r["predicted_code"] = kept
            r["predicted_code_meta"] = meta
            rows_out.append(r)
        if not args.dry_run:
            out_path = os.path.join(args.out_dir, os.path.basename(path))
            with open(out_path, "w") as fh:
                for r in rows_out:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    tot = sum(stat.values())
    print("\n=== per-sample status totals ===")
    for k in ("complete", "spliced", "dropped"):
        print(f"  {k:14}: {stat[k]:7d}  ({100*stat[k]/max(1,tot):.1f}%)")
    print(f"  {'TOTAL':14}: {tot:7d}")
    print("\n=== by source ===")
    for s in sorted({s for s, _ in by_src}):
        c = by_src[(s, 'complete')]; rc = by_src[(s, 'spliced')]; d = by_src[(s, 'dropped')]
        t = c + rc + d
        print(f"  {s:26} complete={c:6d} spliced={rc:5d} dropped={d:5d}  kept={100*(c+rc)/max(1,t):.1f}%")
    if args.dry_run:
        print("\n(dry run: no files written)")


if __name__ == "__main__":
    main()
