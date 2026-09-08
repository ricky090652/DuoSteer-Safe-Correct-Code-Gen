"""
prepare_codeql_seccodeplt.py

Extract baseline generated code from data/res_code_gen_seccodeplt.jsonl,
syntax-validate each snippet, append the CWE-specific CodeQL entry-point wrapper,
and write organised directories ready for CodeQL analysis.

Only processes records whose cwe_id maps to one of the 6 covered CWEs:
  22 → cwe-022, 79 → cwe-079, 94 → cwe-094,
  295 → cwe-295, 327 → cwe-327, 502 → cwe-502

Output layout:
  data/codeql/seccodeplt_baseline/cwe-XXX/<record_id>__gen0.py

Usage:
  python prepare_codeql_seccodeplt.py
  python prepare_codeql_seccodeplt.py --dry_run
"""
# --- repo path setup: allow running this script from any directory ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _d in ("common", "dataset_construction"):
    _p = _REPO_ROOT / _d
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# ---------------------------------------------------------------------

import ast
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

from codeql_entry_points import add_entry_point

BASE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE / "data" / "res_code_gen_seccodeplt.jsonl"
DEFAULT_OUT_ROOT = BASE / "data" / "codeql" / "seccodeplt_baseline"

COVERED_CWES = {
    "22": "cwe-022",
    "79": "cwe-079",
    "94": "cwe-094",
    "295": "cwe-295",
    "327": "cwe-327",
    "502": "cwe-502",
}

_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)\n?```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"^```(?:python)?\s*\n", re.IGNORECASE)


def strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Unclosed fence: strip only the opening ```[python] line
    return _OPEN_FENCE_RE.sub("", text).strip()


def syntax_check(code: str) -> tuple[bool, str]:
    if not code:
        return False, "empty"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SecCodePLT baseline code for CodeQL analysis."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="JSONL with predicted_code (default: Llama baseline)")
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT,
                        help="Output root for per-CWE .py files")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report counts without writing files")
    args = parser.parse_args()

    INPUT = args.input
    OUT_ROOT = args.out_root
    if not INPUT.exists():
        print(f"ERROR: input file not found: {INPUT}", file=sys.stderr)
        sys.exit(1)

    stats = {
        "total": 0,
        "skipped_cwe": 0,
        "invalid": 0,
        "written": 0,
    }
    per_cwe: dict[str, int] = {cwe: 0 for cwe in COVERED_CWES.values()}
    invalid_log: list[dict] = []

    if not args.dry_run:
        for cwe in COVERED_CWES.values():
            (OUT_ROOT / cwe).mkdir(parents=True, exist_ok=True)

    with open(INPUT, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON error at line {lineno}: {e}", file=sys.stderr)
                continue

            cwe_raw = str(record.get("cwe_id", ""))
            if cwe_raw not in COVERED_CWES:
                stats["skipped_cwe"] += 1
                continue

            cwe_id = COVERED_CWES[cwe_raw]
            record_id = record.get("id", f"seccodeplt_{cwe_raw}_{lineno}")
            stats["total"] += 1

            predicted = record.get("predicted_code", [])
            if isinstance(predicted, str):
                predicted = [predicted]

            for gen_idx, raw_code in enumerate(predicted):
                code = strip_fences(raw_code)
                valid, err = syntax_check(code)
                if not valid:
                    stats["invalid"] += 1
                    invalid_log.append({"record_id": record_id, "gen_idx": gen_idx, "error": err})
                    continue

                filename = f"{record_id}__gen{gen_idx}.py"
                wrapped = add_entry_point(code, cwe_id)
                out_path = OUT_ROOT / cwe_id / filename
                if not args.dry_run:
                    out_path.write_text(wrapped, encoding="utf-8")
                stats["written"] += 1
                per_cwe[cwe_id] += 1

    tag = " [DRY RUN]" if args.dry_run else ""
    print(f"\nSecCodePLT baseline CodeQL prep{tag}")
    print(f"  Records in covered CWEs : {stats['total']}")
    print(f"  Records skipped (other CWE): {stats['skipped_cwe']}")
    print(f"  Snippets invalid/empty  : {stats['invalid']}")
    print(f"  Files written           : {stats['written']}")
    for cwe, cnt in sorted(per_cwe.items()):
        print(f"    {cwe}: {cnt:>5} files")

    if invalid_log and not args.dry_run:
        log_path = OUT_ROOT / "invalid_code.jsonl"
        with open(log_path, "w", encoding="utf-8") as fh:
            for entry in invalid_log:
                fh.write(json.dumps(entry) + "\n")
        try:
            rel = log_path.relative_to(BASE)
        except ValueError:
            rel = log_path
        print(f"\n  Invalid log: {rel} ({len(invalid_log)} entries)")

    if not args.dry_run:
        print(f"\nFiles written under: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
