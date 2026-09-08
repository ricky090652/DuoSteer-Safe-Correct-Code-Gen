"""
DS-5a: Prepare GPT-4.1 batch API input for correctness labeling of steered outputs.

Reads codeql_annotated.jsonl (codeql_pass=True records only) and builds
the OpenAI batch JSONL for submission.

Usage:
    python correctness_batch_prepare.py
    python correctness_batch_prepare.py --dry_run
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

import argparse
import json
import re
from pathlib import Path

from prompts import CODE_CORRECTNESS_EVALUATION

FILTERED_BASE = Path("data/double_steering/filtered")
LABEL_BASE    = Path("data/double_steering/correctness_labels")
CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]
MODEL = "gpt-4.1"

MAX_ALPHA = 10.0   # exclude α > 10 (degenerate outputs, out of scope)
MAX_TOPK  = 128    # exclude top_k > 128 (out of scope)


def _alpha_from_condition(condition: str):
    m = re.search(r"alpha(\d+\.?\d*)", condition)
    return float(m.group(1)) if m else None


def _topk_from_condition(condition: str):
    m = re.search(r"top(\d+)_", condition)
    return int(m.group(1)) if m else None


def _in_scope(rec: dict) -> bool:
    """Return True if this record is within the α≤10, top_k≤128 scope."""
    cond = rec.get("condition", "")
    alpha = _alpha_from_condition(cond)
    if alpha is not None and alpha > MAX_ALPHA:
        return False
    topk = _topk_from_condition(cond)
    if topk is not None and topk > MAX_TOPK:
        return False
    return True


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def make_batch_record(rec: dict) -> dict:
    question = rec.get("question", "")
    code = rec.get("extracted_code") or extract_code(rec.get("predicted_code", ""))

    # custom_id encodes enough to reconstruct provenance
    custom_id = f"{rec['cwe_id']}__{rec['config']}__{rec['condition']}__{rec['id']}"

    prompt = CODE_CORRECTNESS_EVALUATION \
        .replace("{PROBLEM}", question) \
        .replace("{CODE}", code)

    return {
        "custom_id": custom_id,
        "method":    "POST",
        "url":       "/v1/chat/completions",
        "body": {
            "model":       MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens":  1024,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--filtered_base", default="data/double_steering/filtered",
                        help="Directory with CodeQL-annotated filtered generations")
    parser.add_argument("--label_base", default="data/double_steering/correctness_labels",
                        help="Directory for batch files and correctness labels")
    args = parser.parse_args()
    global FILTERED_BASE, LABEL_BASE
    FILTERED_BASE = Path(args.filtered_base)
    LABEL_BASE = Path(args.label_base)

    LABEL_BASE.mkdir(parents=True, exist_ok=True)

    for cwe in CWES:
        ann_path = FILTERED_BASE / cwe / "codeql_annotated.jsonl"
        if not ann_path.exists():
            print(f"{cwe}: codeql_annotated.jsonl not found, skipping")
            continue

        records = []
        n_skipped = 0
        with open(ann_path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("codeql_pass"):
                    continue
                if not _in_scope(r):
                    n_skipped += 1
                    continue
                records.append(r)
        if n_skipped:
            print(f"{cwe}: skipped {n_skipped} out-of-scope records (α>10 or top_k>128)")

        out_dir = LABEL_BASE / cwe
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "batch_input_gpt41.jsonl"

        if args.dry_run:
            print(f"{cwe}: {len(records)} records would be written to {out_path}")
            continue

        with open(out_path, "w") as f:
            for rec in records:
                batch_rec = make_batch_record(rec)
                f.write(json.dumps(batch_rec) + "\n")

        print(f"{cwe}: {len(records)} batch records written to {out_path}")


if __name__ == "__main__":
    main()
