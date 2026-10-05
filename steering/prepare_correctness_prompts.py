"""
Extract the unique training tasks per CWE from the contrastive pairs.

For each CWE, reads primary (and supplementary for cwe-094/295/502) contrastive
pair files, deduplicates by question (src_id, or a hash of the question text
when the pair files carry no src_id), and writes task JSONL files for
steer_eval.py (which renders the benign prompt from `question` at run time).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import question_group_id, question_hash

ALL_CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]

# Per-CWE pair file kinds (files are {pair_prefix}_{kind}.jsonl under --pair_dir).
# CWE-022/079 use the intra pair set; CWE-094/295/502 additionally use
# cross-prompt pairs to augment the smaller intra-prompt pools.
# The combined per-model files hold every CWE, so records are filtered by
# cwe_id at load time. --cross_for_all adds cross pairs for every CWE (useful
# when a model's intra pools are small, e.g. Qwen CWE-022/079).
PAIR_FILE_KINDS = {
    "cwe-022": ["intra"],
    "cwe-079": ["intra"],
    "cwe-094": ["intra", "cross"],
    "cwe-295": ["intra", "cross"],
    "cwe-502": ["intra", "cross"],
}


def _cwe_key(cwe_id: str) -> str:
    """Normalize a CWE id for comparison: 'cwe-022', '022', '22' -> '22'."""
    return str(cwe_id).lower().removeprefix("cwe-").lstrip("0") or "0"

# Filled in from CLI args in main().
CWES: list = []
PAIR_FILES: dict = {}
EVAL_FILES: dict = {}
OUT_DIR = Path("data/double_steering/tasks")


def load_eval_keys(cwe: str) -> tuple[set, set]:
    """Return (src_ids, question hashes) of the evaluation set for this CWE."""
    path = Path(EVAL_FILES[cwe])
    if not path.exists():
        return set(), set()
    src_ids, q_hashes = set(), set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("src_id"):
                src_ids.add(r["src_id"])
            if r.get("question"):
                q_hashes.add(question_hash(r["question"]))
    return src_ids, q_hashes


def overlaps_eval(task: dict, eval_src_ids: set, eval_q_hashes: set) -> bool:
    return (task["src_id"] in eval_src_ids
            or question_hash(task["question"]) in eval_q_hashes)


def extract_prompts(cwe: str) -> tuple[list, int]:
    """Return (one task per question, number of pair records read)."""
    seen: dict[str, dict] = {}  # src_id -> task record
    n_records = 0
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
                n_records += 1
                sid = question_group_id(r)
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
    return list(seen.values()), n_records


def main():
    global CWES, PAIR_FILES, EVAL_FILES, OUT_DIR
    ap = argparse.ArgumentParser(
        description="Extract deduplicated training prompts per CWE from "
                    "contrastive pairs, excluding evaluation-set questions.")
    ap.add_argument("--pair_dir", default="data/codesec_pairs",
                    help="Directory containing the contrastive pair JSONL files")
    ap.add_argument("--pair_prefix", default="llama31-8b",
                    help="Pair file prefix, e.g. llama31-8b or qwen25-coder-7b")
    ap.add_argument("--eval_prompt_dir", default="data/eval_tasks",
                    help="Directory with seccodeplt_{cwe}.jsonl eval tasks")
    ap.add_argument("--out_dir", default="data/double_steering/tasks",
                    help="Output directory for train_tasks_{cwe}.jsonl files")
    ap.add_argument("--cwes", nargs="+", default=ALL_CWES, choices=ALL_CWES)
    ap.add_argument("--cross_for_all", action="store_true",
                    help="Draw questions from both intra and cross pairs for every CWE "
                         "(default: cross only for cwe-094/295/502)")
    args = ap.parse_args()

    CWES = args.cwes
    kinds = {cwe: (["intra", "cross"] if args.cross_for_all else PAIR_FILE_KINDS[cwe])
             for cwe in CWES}
    PAIR_FILES = {cwe: [str(Path(args.pair_dir) / f"{args.pair_prefix}_{kind}.jsonl")
                        for kind in kinds[cwe]]
                  for cwe in CWES}
    EVAL_FILES = {cwe: str(Path(args.eval_prompt_dir) / f"seccodeplt_{cwe.replace('-', '')}.jsonl")
                  for cwe in CWES}
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cwe in CWES:
        eval_src_ids, eval_q_hashes = load_eval_keys(cwe)
        prompts, n_records = extract_prompts(cwe)

        # Filter out any question overlapping with the eval set
        before = len(prompts)
        prompts = [p for p in prompts
                   if not overlaps_eval(p, eval_src_ids, eval_q_hashes)]
        filtered = before - len(prompts)

        out_path = OUT_DIR / f"train_tasks_{cwe}.jsonl"
        with open(out_path, "w") as f:
            for p in prompts:
                f.write(json.dumps(p) + "\n")

        print(f"{cwe}: {len(prompts)} unique questions written"
              f"  (from {n_records} pairs; eval overlap removed: {filtered})")

    # Verification: no eval overlap
    print("\n=== Cross-contamination check ===")
    for cwe in CWES:
        eval_src_ids, eval_q_hashes = load_eval_keys(cwe)
        with open(OUT_DIR / f"train_tasks_{cwe}.jsonl") as f:
            tasks = [json.loads(l) for l in f]
        overlap = sum(overlaps_eval(t, eval_src_ids, eval_q_hashes) for t in tasks)
        status = "OK" if not overlap else f"FAIL ({overlap} overlap)"
        print(f"  {cwe}: {status}")


if __name__ == "__main__":
    main()
