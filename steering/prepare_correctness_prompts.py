"""
Extract the unique training tasks per CWE from the contrastive pairs.

For each CWE, reads primary (and supplementary for cwe-094/295/502) contrastive
pair files, deduplicates by src_id, and writes task JSONL files for steer_eval.py
(which renders the benign prompt from `question` at run time).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ALL_CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]

# Per-CWE pair file name patterns (relative to --pair_dir).
# CWE-022/079 use the intra pair set; CWE-094/295/502 additionally use
# cross-prompt pairs to augment the smaller intra-prompt pools.
# The combined per-model files hold every CWE, so records are filtered by
# cwe_id at load time. Point PAIR_PREFIX at the model whose pairs you use.
PAIR_PREFIX = "llama31-8b"
PAIR_FILE_PATTERNS = {
    "cwe-022": [f"{PAIR_PREFIX}_intra.jsonl"],
    "cwe-079": [f"{PAIR_PREFIX}_intra.jsonl"],
    "cwe-094": [f"{PAIR_PREFIX}_intra.jsonl", f"{PAIR_PREFIX}_cross.jsonl"],
    "cwe-295": [f"{PAIR_PREFIX}_intra.jsonl", f"{PAIR_PREFIX}_cross.jsonl"],
    "cwe-502": [f"{PAIR_PREFIX}_intra.jsonl", f"{PAIR_PREFIX}_cross.jsonl"],
}


def _cwe_key(cwe_id: str) -> str:
    """Normalize a CWE id for comparison: 'cwe-022', '022', '22' -> '22'."""
    return str(cwe_id).lower().removeprefix("cwe-").lstrip("0") or "0"

# Filled in from CLI args in main().
CWES: list = []
PAIR_FILES: dict = {}
EVAL_FILES: dict = {}
OUT_DIR = Path("data/double_steering/tasks")


def load_eval_src_ids(cwe: str) -> set:
    path = Path(EVAL_FILES[cwe])
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ids.add(r.get("src_id", ""))
    return ids


def extract_prompts(cwe: str) -> list:
    seen: dict[str, dict] = {}  # src_id -> prompt record
    for fpath in PAIR_FILES[cwe]:
        p = Path(fpath)
        if not p.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if _cwe_key(r.get("cwe_id", "")) != _cwe_key(cwe):
                    continue
                sid = r.get("src_id", r["id"])
                if sid in seen:
                    continue
                # task record only; steer_eval.py renders the benign prompt at run time
                seen[sid] = {
                    "id": r["id"],
                    "cwe_id": cwe,
                    "question": r["question"],
                    "source": r.get("source", ""),
                    "src_id": sid,
                }
    return list(seen.values())


def main():
    global CWES, PAIR_FILES, EVAL_FILES, OUT_DIR
    ap = argparse.ArgumentParser(
        description="Extract deduplicated training prompts per CWE from "
                    "contrastive pairs, excluding evaluation-set questions.")
    ap.add_argument("--pair_dir", default="data/contrastive_pairs",
                    help="Directory containing the contrastive pair JSONL files")
    ap.add_argument("--eval_prompt_dir", default="data/eval_tasks",
                    help="Directory with prompt_seccodeplt_{cwe}.jsonl eval prompts")
    ap.add_argument("--out_dir", default="data/double_steering/tasks",
                    help="Output directory for train_tasks_{cwe}.jsonl files")
    ap.add_argument("--cwes", nargs="+", default=ALL_CWES, choices=ALL_CWES)
    args = ap.parse_args()

    CWES = args.cwes
    PAIR_FILES = {cwe: [str(Path(args.pair_dir) / n) for n in PAIR_FILE_PATTERNS[cwe]]
                  for cwe in CWES}
    EVAL_FILES = {cwe: str(Path(args.eval_prompt_dir) / f"seccodeplt_{cwe.replace('-', '')}.jsonl")
                  for cwe in CWES}
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cwe in CWES:
        eval_ids = load_eval_src_ids(cwe)
        prompts = extract_prompts(cwe)

        # Filter out any src_id overlapping with eval set
        before = len(prompts)
        prompts = [p for p in prompts if p["src_id"] not in eval_ids]
        filtered = before - len(prompts)

        out_path = OUT_DIR / f"train_tasks_{cwe}.jsonl"
        with open(out_path, "w") as f:
            for p in prompts:
                f.write(json.dumps(p) + "\n")

        print(f"{cwe}: {len(prompts)} unique prompts written"
              f"  (eval overlap removed: {filtered})")

    # Verification: no eval src_id overlap
    print("\n=== Cross-contamination check ===")
    for cwe in CWES:
        train_ids = {json.loads(l)["src_id"]
                     for l in open(OUT_DIR / f"train_tasks_{cwe}.jsonl")}
        eval_ids = load_eval_src_ids(cwe)
        overlap = train_ids & eval_ids
        status = "OK" if not overlap else f"FAIL ({len(overlap)} overlap)"
        print(f"  {cwe}: {status}")


if __name__ == "__main__":
    main()
