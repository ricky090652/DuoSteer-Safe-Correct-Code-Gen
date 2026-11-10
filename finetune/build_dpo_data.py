"""
Build DPO preference data (prompt / chosen / rejected) from the pairs DuoSteer uses.

Sources (one pooled dataset over all CWEs):
  safety       intra-prompt CodeSec pairs: chosen = safe_code, rejected = vuln_code
               (the pairs behind the safety MD vectors)
  correctness  correctness pairs: chosen = safe & correct, rejected = safe & incorrect
               (the pairs behind DuoSteer's correctness vectors)
  cross        optional (--include_cross): cross-prompt pairs, whose vuln side was sampled
               under the vulnerability-eliciting prompt. DuoSteer does not use them for
               vectors, so they are off by default.

Every prompt is the benign prompt p^b (common.prompts.CODE_GENERATION_PROMPT), the same
one steer_eval.py renders at evaluation time. Responses are kept verbatim (fenced code),
matching what the model emits.

Split: question-level (common.utils.question_hash) across all sources, so a question never
appears in both train and val. Questions in the SecCodePLT eval tasks are dropped.

Output: {out_dir}/train.jsonl, {out_dir}/val.jsonl, {out_dir}/stats.json

Usage:
  python finetune/build_dpo_data.py \\
      --pair_prefix qwen25-coder-7b \\
      --correctness_dir data/double_steering/qwen_merged/correctness_pairs \\
      --out_dir data/dpo/qwen
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.prompts import CODE_GENERATION_PROMPT, normalize_cwe_id
from common.utils import question_hash, read_jsonl, write_jsonl

ALL_CWES = ["022", "079", "094", "295", "502"]


def cwe3(cwe_id) -> str:
    """'cwe-022' / '22' / 22 -> '022'."""
    return f"{int(normalize_cwe_id(cwe_id)):03d}"


def to_record(pair: dict, source: str) -> dict:
    return {
        "id": pair["id"],
        "source": source,
        "cwe_id": cwe3(pair["cwe_id"]),
        "qid": question_hash(pair["question"]),
        "prompt": CODE_GENERATION_PROMPT.format(question=pair["question"]),
        "chosen": pair["safe_code"],
        "rejected": pair["vuln_code"],
    }


def load_correctness(corr_dir: Path, cwes: list) -> list:
    pairs = []
    for cwe in cwes:
        files = sorted(corr_dir.glob(f"cwe-{cwe}_*.jsonl"))
        if not files:
            raise SystemExit(f"No correctness pairs for CWE-{cwe} in {corr_dir}")
        for f in files:
            pairs.extend(read_jsonl(f))
    return pairs


def eval_question_hashes(eval_dir: Path) -> set:
    hashes = set()
    for f in sorted(eval_dir.glob("seccodeplt_cwe*.jsonl")):
        hashes.update(question_hash(r["question"]) for r in read_jsonl(f))
    return hashes


def main(args):
    cwes = [cwe3(c) for c in args.cwes]
    pair_dir = Path(args.pair_dir)

    records = []
    intra = read_jsonl(pair_dir / f"{args.pair_prefix}_intra.jsonl")
    records += [to_record(p, "safety") for p in intra]
    if args.correctness_dir:
        records += [to_record(p, "correctness")
                    for p in load_correctness(Path(args.correctness_dir), cwes)]
    if args.include_cross:
        cross = read_jsonl(pair_dir / f"{args.pair_prefix}_cross.jsonl")
        records += [to_record(p, "cross") for p in cross]
    records = [r for r in records if r["cwe_id"] in cwes]
    n_loaded = len(records)

    # Drop degenerate and duplicate triples.
    seen, kept = set(), []
    for r in records:
        key = (r["prompt"], r["chosen"], r["rejected"])
        if r["chosen"].strip() == r["rejected"].strip() or key in seen:
            continue
        seen.add(key)
        kept.append(r)
    n_dup = n_loaded - len(kept)

    eval_hashes = eval_question_hashes(Path(args.eval_dir))
    before = len(kept)
    kept = [r for r in kept if r["qid"] not in eval_hashes]
    n_eval_overlap = before - len(kept)

    # Question-level split, stratified by CWE (a question belongs to the CWE of its first pair).
    q_cwe = {}
    for r in kept:
        q_cwe.setdefault(r["qid"], r["cwe_id"])
    by_cwe = defaultdict(list)
    for q, c in q_cwe.items():
        by_cwe[c].append(q)
    rng = random.Random(args.seed)
    val_q = set()
    for c in sorted(by_cwe):
        qs = sorted(by_cwe[c])
        rng.shuffle(qs)
        val_q.update(qs[:max(1, round(len(qs) * args.val_ratio))])

    train = [r for r in kept if r["qid"] not in val_q]
    val = [r for r in kept if r["qid"] in val_q]
    rng.shuffle(train)
    assert not ({r["qid"] for r in train} & {r["qid"] for r in val}), "question leak"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")

    def count(rs):
        return {f"{s}/{c}": n for (s, c), n in sorted(Counter((r["source"], r["cwe_id"]) for r in rs).items())}

    stats = {
        "args": vars(args),
        "n_loaded": n_loaded,
        "n_dropped_dup_or_identical": n_dup,
        "n_dropped_eval_overlap": n_eval_overlap,
        "n_train": len(train),
        "n_val": len(val),
        "n_train_questions": len({r["qid"] for r in train}),
        "n_val_questions": len(val_q),
        "train_by_source_cwe": count(train),
        "val_by_source_cwe": count(val),
    }
    json.dump(stats, open(out / "stats.json", "w"), indent=2)

    print(f"loaded {n_loaded}, dropped {n_dup} dup/identical, {n_eval_overlap} eval-overlap")
    print(f"train {len(train)} pairs / {stats['n_train_questions']} questions, "
          f"val {len(val)} pairs / {len(val_q)} questions")
    for k, n in stats["train_by_source_cwe"].items():
        print(f"  train {k:<18} {n:>5}   val {stats['val_by_source_cwe'].get(k, 0):>4}")
    print(f"wrote {out}/train.jsonl, val.jsonl, stats.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair_dir", default="data/codesec_pairs")
    ap.add_argument("--pair_prefix", default="qwen25-coder-7b")
    ap.add_argument("--correctness_dir", default="data/double_steering/qwen_merged/correctness_pairs",
                    help="Correctness pair dir ({cwe-XXX}_{train,val}.jsonl); empty string to skip.")
    ap.add_argument("--include_cross", action="store_true")
    ap.add_argument("--eval_dir", default="data/eval_tasks")
    ap.add_argument("--cwes", nargs="+", default=ALL_CWES)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data/dpo/qwen")
    main(ap.parse_args())
