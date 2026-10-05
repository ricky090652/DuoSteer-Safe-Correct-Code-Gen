"""
DS-6: Build correctness contrastive pairs from GPT-4.1 labeled records.

Pairing rules:
  - Incorrect pool per question: deduplicated by response hash, each used exactly once.
  - Correct pool per question: deduplicated; cycled in shuffled order when
    n_incorrect > n_correct (avoids always reusing the same record).
  - Global pair cap per CWE = PAIR_TARGETS[cwe]; applied via round-robin across
    src_ids to maximize question-level diversity.
  - 80/20 train/val split at question (src_id) level.

Pair targets match safety steering pair counts (min 400):
  cwe-022=541, cwe-079=688, cwe-094=867, cwe-295=644, cwe-502=722
Override with --pair_target (e.g. 300 for Qwen-2.5-Coder, as in the paper).

Every labeled record must carry a question-level src_id (written by
prepare_correctness_prompts.py); there is no fallback to the record id.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ALL_CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]

# Filled in from CLI args in main().
LABEL_BASE = Path("data/double_steering/correctness_labels")
OUT_DIR    = Path("data/double_steering/correctness_contrastive_pairs")
CWES = ALL_CWES
VAL_RATIO  = 0.20
SEED       = 42

# Match safety steering pair counts; all >= 400
PAIR_TARGETS = {
    "cwe-022": 541,
    "cwe-079": 688,
    "cwe-094": 867,
    "cwe-295": 644,
    "cwe-502": 722,
}


def resp_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()


def dedup(recs: list, key: str = "predicted_code") -> list:
    seen: set = set()
    out = []
    for r in recs:
        h = resp_hash(r.get(key, ""))
        if h not in seen:
            seen.add(h)
            out.append(r)
    return out


def load_labeled(cwe: str) -> list:
    path = LABEL_BASE / cwe / "labeled.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("codeql_pass") and r.get("gpt41_correct") is not None:
                records.append(r)
    missing = sum(1 for r in records if not r.get("src_id"))
    if missing:
        raise ValueError(
            f"{cwe}: {missing} labeled records have no src_id; grouping by record id "
            "would split questions across pairs and leak them into val. Regenerate "
            "the tasks with prepare_correctness_prompts.py and rerun the pipeline.")
    return records


def _apply_cap_roundrobin(
    pairs_by_sid: dict[str, list],
    target: int,
    rng: random.Random,
) -> list:
    """Take at most `target` pairs using round-robin across src_ids for diversity."""
    queues = {sid: list(pairs) for sid, pairs in pairs_by_sid.items()}
    sid_order = list(queues.keys())
    rng.shuffle(sid_order)

    result = []
    while len(result) < target:
        added_any = False
        for sid in sid_order:
            if queues[sid]:
                result.append(queues[sid].pop(0))
                added_any = True
                if len(result) >= target:
                    break
        if not added_any:
            break
    return result


def build_pairs(records: list, cwe: str, rng: random.Random) -> list:
    target = PAIR_TARGETS[cwe]

    by_q: dict[str, dict] = defaultdict(lambda: {"correct": [], "incorrect": []})
    for rec in records:
        sid = rec["src_id"]
        pool = "correct" if rec["gpt41_correct"] else "incorrect"
        by_q[sid][pool].append(rec)

    pairs_by_sid: dict[str, list] = {}

    for sid, pools in by_q.items():
        correct_recs   = pools["correct"]
        incorrect_recs = pools["incorrect"]

        if not correct_recs or not incorrect_recs:
            continue

        def _alpha(r):
            try:
                return float(r.get("alpha") or 0)
            except (TypeError, ValueError):
                return 0.0

        # Correct: sort by alpha ascending (low α = closest to baseline = highest quality),
        # break ties randomly to diversify across configs/conditions within same alpha
        correct_recs = sorted(correct_recs,
                              key=lambda r: (_alpha(r), rng.random()))
        correct_recs = dedup(correct_recs)

        # Incorrect: sort by alpha descending (high α = strongest contrast)
        incorrect_recs = sorted(incorrect_recs,
                                key=lambda r: -_alpha(r))
        incorrect_recs = dedup(incorrect_recs)

        n_correct   = len(correct_recs)
        n_incorrect = len(incorrect_recs)

        # Shuffled cycle indices: when n_incorrect > n_correct, cycle correct in
        # random order, so the same record is not always used first
        shuffled_idx = list(range(n_correct))
        rng.shuffle(shuffled_idx)

        q_pairs = []
        for i, inc_rec in enumerate(incorrect_recs):
            cor_rec = correct_recs[shuffled_idx[i % n_correct]]
            # Skip pairs where both sides are identical code (contradictory GPT labels)
            if resp_hash(cor_rec.get("predicted_code", "")) == resp_hash(inc_rec.get("predicted_code", "")):
                continue
            pair_id = f"ds_{cwe}_{sid}_pair{i}"
            # Flat schema: safe_code is the safe-and-correct generation,
            # vuln_code the safe-but-incorrect one (both CodeQL-safe; the
            # contrast isolates functional correctness). The *_config /
            # *_condition fields record which steering run produced each side.
            pair = {
                "id":             pair_id,
                "cwe_id":         cwe,
                "question":       inc_rec.get("question", ""),
                "source":         "double_steering_phase1",
                "src_id":         sid,
                "safe_code":      cor_rec.get("predicted_code", ""),
                "vuln_code":      inc_rec.get("predicted_code", ""),
                "safe_config":    cor_rec.get("config", ""),
                "safe_condition": cor_rec.get("condition", ""),
                "vuln_config":    inc_rec.get("config", ""),
                "vuln_condition": inc_rec.get("condition", ""),
            }
            q_pairs.append(pair)

        rng.shuffle(q_pairs)
        pairs_by_sid[sid] = q_pairs

    return _apply_cap_roundrobin(pairs_by_sid, target, rng)


def split_pairs(pairs: list) -> tuple[list, list]:
    """Question-level 80/20 split by src_id."""
    rng = random.Random(SEED + 1)
    src_ids = list({p["src_id"] for p in pairs})
    rng.shuffle(src_ids)
    n_val = max(1, int(len(src_ids) * VAL_RATIO))
    val_ids = set(src_ids[:n_val])
    train = [p for p in pairs if p["src_id"] not in val_ids]
    val   = [p for p in pairs if p["src_id"] in val_ids]
    return train, val


def verify_pairs(pairs: list, cwe: str, split: str):
    by_q: dict[str, list] = defaultdict(list)
    for p in pairs:
        by_q[p["src_id"]].append(p["vuln_code"].strip())
    dup_qs = {q for q, resps in by_q.items()
              if len(resps) != len(set(map(resp_hash, resps)))}
    if dup_qs:
        print(f"  WARNING {cwe} {split}: {len(dup_qs)} questions have duplicate"
              " incorrect code — check ds6 logic")

    by_q2: dict[str, dict] = defaultdict(lambda: {"correct": [], "incorrect": []})
    for p in pairs:
        by_q2[p["src_id"]]["correct"].append(p["safe_code"].strip())
        by_q2[p["src_id"]]["incorrect"].append(p["vuln_code"].strip())
    bad = 0
    for q, d in by_q2.items():
        n_uniq_c = len(set(map(resp_hash, d["correct"])))
        n_inc    = len(d["incorrect"])
        if len(d["correct"]) != n_uniq_c and n_uniq_c >= n_inc:
            bad += 1
    if bad:
        print(f"  WARNING {cwe} {split}: {bad} questions with unnecessary correct reuse")


def main():
    global LABEL_BASE, OUT_DIR, CWES, VAL_RATIO, SEED, PAIR_TARGETS
    ap = argparse.ArgumentParser(
        description="Build (safe-and-correct, safe-but-incorrect) correctness "
                    "contrastive pairs from GPT-4.1 labeled steered generations.")
    ap.add_argument("--label_base", default="data/double_steering/correctness_labels",
                    help="Directory with {cwe}/labeled.jsonl from the correctness batch step")
    ap.add_argument("--out_dir", default="data/double_steering/correctness_contrastive_pairs",
                    help="Output directory for {cwe}_train.jsonl / {cwe}_val.jsonl")
    ap.add_argument("--cwes", nargs="+", default=ALL_CWES, choices=ALL_CWES)
    ap.add_argument("--val_ratio", type=float, default=0.20,
                    help="Question-level validation split ratio")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pair_target", type=int, default=None,
                    help="Pair cap for every CWE (default: per-CWE PAIR_TARGETS)")
    args = ap.parse_args()
    LABEL_BASE = Path(args.label_base)
    OUT_DIR = Path(args.out_dir)
    CWES = args.cwes
    VAL_RATIO = args.val_ratio
    SEED = args.seed
    if args.pair_target is not None:
        PAIR_TARGETS = {cwe: args.pair_target for cwe in ALL_CWES}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    for cwe in CWES:
        records = load_labeled(cwe)
        if not records:
            print(f"{cwe}: no labeled records — skipping (run DS-5 first)")
            continue

        target = PAIR_TARGETS[cwe]
        n_src     = len({r["src_id"] for r in records})
        n_correct = sum(1 for r in records if r["gpt41_correct"])
        n_incorr  = sum(1 for r in records if not r["gpt41_correct"])
        print(f"{cwe}: {len(records)} labeled records  "
              f"(correct={n_correct} incorrect={n_incorr} src_ids={n_src})  "
              f"target={target}")

        pairs = build_pairs(records, cwe, rng)
        train, val = split_pairs(pairs)

        for split, data in [("train", train), ("val", val)]:
            out = OUT_DIR / f"{cwe}_{split}.jsonl"
            with open(out, "w") as f:
                for p in data:
                    f.write(json.dumps(p) + "\n")
            verify_pairs(data, cwe, split)

        total = len(train) + len(val)
        pct   = 100 * total / target
        ok    = "OK" if total >= int(target * 0.9) else f"LOW ({pct:.0f}% of target)"
        print(f"  train={len(train)}  val={len(val)}  total={total}/{target}  [{ok}]")

        tr_ids = {p["src_id"] for p in train}
        va_ids = {p["src_id"] for p in val}
        leak   = tr_ids & va_ids
        if leak:
            print(f"  WARNING: {len(leak)} src_id appear in both splits!")


if __name__ == "__main__":
    main()
