"""
Build held-out task files for selecting steering configurations (paper App. I).

Held-out questions per CWE:
  - val:        the probe validation questions, i.e. the same question-level
                split as train_probe.make_split_indices (seed 42, 20%), read
                from the intra representation metadata.
  - cross_only: with --add_cross_only, questions that appear only in the
                cross pairs (never used for probes or safety vectors).
Questions overlapping the SecCodePLT evaluation set are dropped. One task per
question (generation is greedy, so repeated questions add nothing).

Output: {out_dir}/heldout_{cwe}.jsonl (e.g. heldout_cwe022.jsonl), in the same
format as the eval task files, so steer_eval.py / duosteer_eval.py,
codeql_eval_steered_modes.py and eval_correctness_steered.py run on it as-is.

Usage:
  python steering/prepare_heldout_tasks.py --pair_prefix qwen25-coder-7b \\
      --metadata_dir data/representations/Qwen_Qwen2_5-Coder-7B-Instruct/qwen25-coder-7b_intra \\
      --out_dir data/eval_tasks/heldout_qwen --add_cross_only
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import question_group_id, question_hash

ALL_CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]


def probe_val_questions(metadata_path: Path, val_ratio: float, seed: int) -> set:
    """Validation src_ids, replicating train_probe.make_split_indices (kept torch-free)."""
    meta = json.load(open(metadata_path))
    question_to_indices = defaultdict(list)
    for pair in meta["pairs"]:
        question_to_indices[pair["src_id"]].append(pair["index"])
    question_ids = list(question_to_indices.keys())
    random.Random(seed).shuffle(question_ids)
    n_val = max(1, round(len(question_ids) * val_ratio))
    return set(question_ids[:n_val])


def load_questions(path: Path, cwe_num: str) -> dict:
    """src_id -> first pair record of that question, for one CWE."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["cwe_id"] == cwe_num:
                out.setdefault(question_group_id(r), r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pair_dir", default="data/codesec_pairs")
    ap.add_argument("--pair_prefix", default="qwen25-coder-7b")
    ap.add_argument("--metadata_dir", required=True,
                    help="Intra representation dir holding {cwe}/response_mean/metadata.json")
    ap.add_argument("--eval_prompt_dir", default="data/eval_tasks")
    ap.add_argument("--out_dir", default="data/eval_tasks/heldout_qwen")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--add_cross_only", action="store_true",
                    help="Also add questions that appear only in the cross pairs")
    ap.add_argument("--cwes", nargs="+", default=ALL_CWES, choices=ALL_CWES)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cwe in args.cwes:
        num = cwe.split("-")[1]
        tag = cwe.replace("-", "")
        intra = load_questions(Path(args.pair_dir) / f"{args.pair_prefix}_intra.jsonl", num)
        cross = load_questions(Path(args.pair_dir) / f"{args.pair_prefix}_cross.jsonl", num)
        val = probe_val_questions(Path(args.metadata_dir) / cwe / "response_mean" / "metadata.json",
                                  args.val_ratio, args.seed)
        missing = val - set(intra)
        if missing:
            raise SystemExit(f"{cwe}: {len(missing)} validation questions not in the intra pairs; "
                             "metadata and pair files do not match")

        picked = [(sid, intra[sid], "val") for sid in sorted(val)]
        if args.add_cross_only:
            picked += [(sid, cross[sid], "cross_only") for sid in sorted(set(cross) - set(intra))]

        eval_hashes = {question_hash(json.loads(l)["question"])
                       for l in open(Path(args.eval_prompt_dir) / f"seccodeplt_{tag}.jsonl")}
        tasks = [{"id": r["id"], "cwe_id": num, "question": r["question"], "source": r.get("source", ""),
                  "src_id": sid, "split": split}
                 for sid, r, split in picked if question_hash(r["question"]) not in eval_hashes]

        out_path = out_dir / f"heldout_{tag}.jsonl"
        with open(out_path, "w") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        n_val = sum(t["split"] == "val" for t in tasks)
        print(f"{cwe}: {len(tasks)} tasks (val {n_val}, cross_only {len(tasks) - n_val}, "
              f"dropped for eval overlap {len(picked) - len(tasks)}) -> {out_path}")


if __name__ == "__main__":
    main()
