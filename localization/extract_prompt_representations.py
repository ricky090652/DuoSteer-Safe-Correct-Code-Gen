"""
Extract prompt-only representations for the CWE router (dynamic CWE routing).

Builds two item sets and runs the model on the user prompt only (no code):
  train  unique questions from the released CodeSec-Pairs files, labelled with
         their CWE. --pair_prefixes is a priority list: a question that occurs
         in several models' pair files takes its CWE set from the first prefix
         that contains it (EM labels come from that model's CodeQL detections,
         so they differ between models). Questions with more than one CWE are
         dropped, as are questions that also appear in the eval set.
  eval   the SecCodePLT tasks in --eval_dir (cwe_id is used for scoring only).

Each prompt is rendered exactly as at steering/eval time:
  CODE_GENERATION_PROMPT + chat template with add_generation_prompt=True.
The representation is the residual stream at the last prompt token (the
position that predicts the first generated token) for every layer.

Output  {output_dir}/{model_slug}/router_prompts/
  train.pt / eval.pt   {"reps": Tensor(n_items, n_layers, hidden_dim)}
                       layer index i in the tensor = transformer layer i+1
  train.jsonl / eval.jsonl
                       one record per row: src_id, question, cwe_id, source,
                       label_origin (train only)
  metadata.json        run config and per-CWE counts

Usage:
  python localization/extract_prompt_representations.py \\
      --model Qwen/Qwen2.5-Coder-7B-Instruct \\
      --pair_prefixes qwen25-coder-7b llama31-8b
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "localization"))
from common.prompts import CODE_GENERATION_PROMPT
from common.utils import question_group_id, question_hash, read_jsonl, write_jsonl
from extract_representations import load_model, slugify

CWES = ["022", "079", "094", "295", "502"]


def norm_cwe(x) -> str:
    """'79' / 'cwe-079' / 79 -> '079'."""
    digits = "".join(ch for ch in str(x) if ch.isdigit()).lstrip("0")
    return digits.zfill(3) if digits else "000"


# --------------------------------------------------------------------------- #
# Item sets
# --------------------------------------------------------------------------- #

def load_eval_items(eval_dir: Path) -> list[dict]:
    items = []
    for cwe in CWES:
        for rec in read_jsonl(eval_dir / f"seccodeplt_cwe{cwe}.jsonl"):
            items.append({
                "src_id": question_group_id(rec),
                "question": rec["question"],
                "cwe_id": norm_cwe(rec["cwe_id"]),
                "source": rec.get("source", "seccodeplt"),
            })
    return items


def build_train_items(pair_dir: Path, prefixes: list[str],
                      eval_hashes: set[str]) -> tuple[list[dict], dict]:
    """Unique single-CWE questions; earlier prefixes win label conflicts."""
    labels: dict[str, set[str]] = {}
    origin: dict[str, str] = {}
    source: dict[str, str] = {}
    for prefix in prefixes:
        q2c = defaultdict(set)
        for kind in ("intra", "cross"):
            fp = pair_dir / f"{prefix}_{kind}.jsonl"
            for rec in read_jsonl(fp):
                q2c[rec["question"]].add(norm_cwe(rec["cwe_id"]))
                source.setdefault(rec["question"], rec.get("source", ""))
        for q, cwes in q2c.items():
            if q not in labels:
                labels[q], origin[q] = cwes, prefix

    stats = Counter()
    items = []
    for q in sorted(labels):
        if question_hash(q) in eval_hashes:
            stats["dropped_eval_overlap"] += 1
            continue
        if len(labels[q]) != 1:
            stats["dropped_multi_cwe"] += 1
            continue
        items.append({
            "src_id": question_hash(q),
            "question": q,
            "cwe_id": next(iter(labels[q])),
            "source": source[q],
            "label_origin": origin[q],
        })
    return items, dict(stats)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def prompt_ids(question: str, tokenizer) -> list[int]:
    messages = [{"role": "user", "content": CODE_GENERATION_PROMPT.format(question=question)}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return list(ids["input_ids"]) if isinstance(ids, dict) or hasattr(ids, "keys") else list(ids)


@torch.no_grad()
def extract(items: list[dict], model, tokenizer, batch_size: int,
            save_dtype: torch.dtype) -> torch.Tensor:
    """Last-prompt-token hidden state per layer -> (n_items, n_layers, hidden)."""
    ids = [prompt_ids(it["question"], tokenizer) for it in items]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))   # length-bucketed batches
    n_layers, hidden = model.config.num_hidden_layers, model.config.hidden_size
    out = torch.empty(len(ids), n_layers, hidden, dtype=save_dtype)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        max_len = max(len(ids[i]) for i in idx)
        # Left padding: position -1 is the last real prompt token for every row.
        input_ids = torch.tensor([[pad_id] * (max_len - len(ids[i])) + ids[i] for i in idx],
                                 device=device)
        attn = torch.tensor([[0] * (max_len - len(ids[i])) + [1] * len(ids[i]) for i in idx],
                            device=device)
        hs = model(input_ids=input_ids, attention_mask=attn,
                   output_hidden_states=True).hidden_states       # n_layers+1 tensors
        last = torch.stack([h[:, -1, :] for h in hs[1:]], dim=1)   # skip embeddings
        out[idx] = last.to("cpu", save_dtype)
        print(f"  {min(b + batch_size, len(order))}/{len(order)}", end="\r")
    print()
    return out


def main(args):
    pair_dir, eval_dir = Path(args.pair_dir), Path(args.eval_dir)
    out_dir = Path(args.output_dir) / slugify(args.model) / "router_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_items = load_eval_items(eval_dir)
    eval_hashes = {question_hash(it["question"]) for it in eval_items}
    train_items, drop_stats = build_train_items(pair_dir, args.pair_prefixes, eval_hashes)
    if args.max_items:
        train_items, eval_items = train_items[:args.max_items], eval_items[:args.max_items]

    train_counts = Counter(it["cwe_id"] for it in train_items)
    origin_counts = Counter((it["label_origin"], it["cwe_id"]) for it in train_items)
    print(f"train: {len(train_items)} questions  {dict(sorted(train_counts.items()))}  "
          f"dropped: {drop_stats}")
    for prefix in args.pair_prefixes:
        print(f"  origin {prefix}: "
              f"{ {c: origin_counts[(prefix, c)] for c in CWES} }")
    print(f"eval:  {len(eval_items)} tasks  "
          f"{dict(sorted(Counter(it['cwe_id'] for it in eval_items).items()))}")

    print(f"Loading model: {args.model} (dtype={args.dtype}, device={args.device}) ...")
    model, tokenizer = load_model(args.model, args.dtype, args.device)
    save_dtype = {"float32": torch.float32, "float16": torch.float16,
                  "bfloat16": torch.bfloat16}[args.dtype]

    for name, items in (("train", train_items), ("eval", eval_items)):
        print(f"Extracting {name} ({len(items)} prompts) ...")
        reps = extract(items, model, tokenizer, args.batch_size, save_dtype)
        torch.save({"reps": reps}, out_dir / f"{name}.pt")
        write_jsonl(items, out_dir / f"{name}.jsonl")
        print(f"  saved {tuple(reps.shape)} -> {out_dir / f'{name}.pt'}")

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "pair_prefixes": args.pair_prefixes,
        "position": "last prompt token (chat template, add_generation_prompt=True)",
        "layer_index": "tensor index i = transformer layer i+1 (embeddings excluded)",
        "n_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "train_counts": dict(sorted(train_counts.items())),
        "train_dropped": drop_stats,
        "eval_counts": dict(sorted(Counter(it["cwe_id"] for it in eval_items).items())),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--pair_dir", default=str(BASE / "data" / "codesec_pairs"))
    p.add_argument("--pair_prefixes", nargs="+", default=["qwen25-coder-7b", "llama31-8b"],
                   help="Pair file prefixes in label-priority order.")
    p.add_argument("--eval_dir", default=str(BASE / "data" / "eval_tasks"))
    p.add_argument("--output_dir", default=str(BASE / "data" / "representations"))
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_items", type=int, default=None,
                   help="Debug: cap both item sets.")
    main(p.parse_args())
