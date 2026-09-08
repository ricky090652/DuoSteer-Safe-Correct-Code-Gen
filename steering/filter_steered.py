"""
DS-3: Three-stage code quality filter on steered generation outputs.

Filters:
  1. Code block presence (triple-backtick fence)
  2. Minimum code length >= 50 characters after extraction
  3. ast.parse() success (syntactic validity)

Reads:  data/double_steering/steered_raw/cwe-{id}/{config}/
Writes: data/double_steering/filtered/cwe-{id}/{config}/ (filtered JSONL)
        data/double_steering/filtered/filter_summary.json
"""

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

ALL_CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]

# Filled in from CLI args in main().
RAW_BASE  = Path("data/double_steering/steered_raw")
OUT_BASE  = Path("data/double_steering/filtered")
SUMMARY_F = OUT_BASE / "filter_summary.json"
CWES = ALL_CWES
MIN_CODE_LEN = 50


def extract_code(text: str) -> str:
    """Extract code from a fenced block; return raw text if no fence found."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: strip markdown-style prefixes
    return text.strip()


def filter_record(record: dict) -> tuple[str, str]:
    """Return (status, extracted_code). Status is 'pass' or reason string."""
    raw = record.get("predicted_code", "")
    if not raw or not raw.strip():
        return "no_code_block", ""

    # Filter 1: must have a code fence
    if "```" not in raw:
        return "no_code_block", ""

    code = extract_code(raw)

    # Filter 2: minimum length
    if len(code) < MIN_CODE_LEN:
        return "too_short", code

    # Filter 3: syntactic validity
    try:
        ast.parse(code)
    except SyntaxError:
        return "syntax_error", code

    return "pass", code


def process_file(in_path: Path, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            status, code = filter_record(rec)
            counts[status] += 1
            rec["filter_status"] = status
            if status == "pass":
                rec["extracted_code"] = code
            fout.write(json.dumps(rec) + "\n")

    n_input = sum(counts.values())
    n_pass  = counts["pass"]
    return {
        "n_input":       n_input,
        "n_pass":        n_pass,
        "n_no_code_block": counts["no_code_block"],
        "n_too_short":   counts["too_short"],
        "n_syntax_error": counts["syntax_error"],
        "pass_rate":     round(n_pass / n_input, 4) if n_input else 0,
    }


def main():
    global RAW_BASE, OUT_BASE, SUMMARY_F, CWES, MIN_CODE_LEN
    ap = argparse.ArgumentParser(
        description="Three-stage quality filter (code fence, minimum length, "
                    "ast.parse) on steered generation outputs.")
    ap.add_argument("--raw_base", default="data/double_steering/steered_raw",
                    help="Directory with raw steered generations: {cwe}/{config}/*.jsonl")
    ap.add_argument("--out_base", default="data/double_steering/filtered",
                    help="Output directory (mirrors the input layout)")
    ap.add_argument("--cwes", nargs="+", default=ALL_CWES, choices=ALL_CWES)
    ap.add_argument("--min_code_len", type=int, default=50,
                    help="Minimum extracted-code length in characters")
    args = ap.parse_args()
    RAW_BASE = Path(args.raw_base)
    OUT_BASE = Path(args.out_base)
    SUMMARY_F = OUT_BASE / "filter_summary.json"
    CWES = args.cwes
    MIN_CODE_LEN = args.min_code_len

    summary = {}

    for cwe in CWES:
        raw_cwe = RAW_BASE / cwe
        if not raw_cwe.exists():
            print(f"  {cwe}: raw dir not found, skipping")
            continue

        cwe_total_pass = 0
        for config_dir in sorted(raw_cwe.iterdir()):
            if not config_dir.is_dir():
                continue
            config = config_dir.name
            for in_file in sorted(config_dir.glob("*.jsonl")):
                rel = f"{cwe}/{config}/{in_file.name}"
                out_file = OUT_BASE / cwe / config / in_file.name
                stats = process_file(in_file, out_file)
                summary[rel] = stats
                cwe_total_pass += stats["n_pass"]
                print(f"  {rel}: {stats['n_pass']}/{stats['n_input']} "
                      f"pass ({100*stats['pass_rate']:.1f}%)")

        print(f"{cwe}: total valid records = {cwe_total_pass}")

    SUMMARY_F.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_F, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {SUMMARY_F}")


if __name__ == "__main__":
    main()
