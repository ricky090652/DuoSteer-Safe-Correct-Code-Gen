"""
DS-9: Double steering evaluation — Methods A, C, D, E.

Four methods for combining safety and correctness activation steering:

  Method A — Independent positions (additive, orthogonalized correctness vector)
      Safety vector v_s on safety-causal top_k1 heads +
      Orthogonalized correctness vector v_c⊥ on correctness-causal top_k2 heads.
      At overlapping heads both offsets accumulate.

  Method C — Correctness-only steering (raw correctness vector, no safety vector)
      Apply raw (non-orthogonalized) v_c on correctness-causal top_k heads.
      Tests whether the correctness direction implicitly encodes safety.

  Method D — Joint single vector (safe+correct vs vuln+incorrect)
      A single vector v_j = mean(h_SC) − mean(h_VI) trained on paired
      (safe+correct, vuln+incorrect) examples.  Applied at safety-causal top_k
      heads with a single α parameter.  No orthogonalization needed because
      v_j is not decomposed into two separate components.

  Method E — Raw-correctness ablation (non-orthogonalized, additive)
      Same architecture as Method A, but uses the raw v_c instead of v_c⊥.
      Since the correctness pairs were built from CodeQL-safe code only,
      v_c is hypothesized to be nearly orthogonal to v_s already.
      This ablation tests whether explicit Gram-Schmidt orthogonalization
      provides measurable benefit.

Also generates baseline (no steering) and safety_only conditions.

α sweep (all methods): [1, 1.5, 2, 3, 5]
top_k sweep (all methods): [8, 16, 32, 64, 128]

Usage:
    python duosteer_eval.py --method D --cwe cwe-022 \\
        --input_file data/prompt_seccodeplt_cwe022.jsonl \\
        --output_dir results/double_steering/eval/cwe-022/method_D \\
        --safety_causal_results data/causal_analysis/cwe-022/head_causal_results.json \\
        --safety_sv_dir <safety_steering_vectors_dir>/cwe-022/response_mean/head/mean_diff \\
        --correct_causal_results results/double_steering/causal_analysis/cwe-022/head_causal_results.json \\
        --correct_sv_dir_orth results/double_steering/steering_vectors/cwe-022/orthogonal/head \\
        --correct_sv_dir_raw  results/double_steering/steering_vectors/cwe-022/raw/head/mean_diff \\
        --joint_sv_dir results/double_steering/steering_vectors/cwe-022/joint/head/mean_diff \\
        --alpha_list 1.0 1.5 2.0 3.0 5.0 --topk_list 8 16 32 64 128 \\
        --model meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --device cuda
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from steer_eval import (
    SteerHead, load_head_sv, load_head_results, select_safe_heads,
    build_prompt, generate_one, get_head_config, load_model,
)


# --------------------------------------------------------------------------- #
# Output filename conventions
# --------------------------------------------------------------------------- #

def make_filename(method: str, alpha1: float, stk: int,
                  alpha2: float, ctk: int, alpha: float, tk: int,
                  jtk: int = 0) -> str:
    if method == "A":
        return f"A_s{alpha1:.1f}_stk{stk}_c{alpha2:.1f}_ctk{ctk}.jsonl"
    if method == "B":
        return f"B_s{alpha1:.1f}_c{alpha2:.1f}_jtk{jtk}.jsonl"
    if method == "C":
        return f"C_c{alpha2:.1f}_ctk{ctk}.jsonl"
    if method == "D":
        return f"D_a{alpha:.1f}_tk{tk}.jsonl"
    if method == "E":
        return f"E_s{alpha1:.1f}_stk{stk}_c{alpha2:.1f}_ctk{ctk}.jsonl"
    raise ValueError(f"Unknown method: {method!r}")


# --------------------------------------------------------------------------- #
# Per-condition generation
# --------------------------------------------------------------------------- #

def run_condition(items, model, tokenizer, hooks_fn, output_file: Path, args):
    done_ids = set()
    existing = []
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "predicted_code" in item:
                    done_ids.add(item["id"])
                    existing.append(item)
        if done_ids:
            print(f"  resume: {len(done_ids)} items done")

    pending = [it for it in items if it["id"] not in done_ids]
    print(f"  {len(pending)} items to generate  →  {output_file.name}")
    if not pending:
        return

    hooks, steering_info = hooks_fn()
    results = list(existing)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        n_total = len(items)
        for item in pending:
            prompt   = build_prompt(item, tokenizer)
            response = generate_one(model, tokenizer, prompt, args)
            results.append({**item, "predicted_code": response,
                            "steering_info": steering_info})
            with open(output_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  [{len(results)}/{n_total}] {item['id']}", end="\r")
    finally:
        for h in hooks:
            h.remove()

    print(f"\n  done — {len(results)} items  →  {output_file}")


# --------------------------------------------------------------------------- #
# Shared baseline / safety-only
# --------------------------------------------------------------------------- #

def run_baseline(items, model, tokenizer, output_dir, args):
    def mk(): return [], {"method": "baseline", "alpha1": 0, "alpha2": 0}
    run_condition(items, model, tokenizer, mk, output_dir / "baseline.jsonl", args)


def run_safety_only(items, model, tokenizer, output_dir, args,
                    safety_ranked, head_dim, alpha1, stk):
    fname = f"safety_only_stk{stk}_a{alpha1:.1f}.jsonl"

    def mk():
        targets = select_safe_heads(safety_ranked, stk)
        hooks = []
        for t in targets:
            sv, _, _ = load_head_sv(args.safety_sv_dir, t["layer"], t["head"])
            hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, alpha1))
        info = {"method": "safety_only", "alpha1": alpha1, "safety_topk": stk}
        return hooks, info

    run_condition(items, model, tokenizer, mk, output_dir / fname, args)


# --------------------------------------------------------------------------- #
# Method A: independent positions, orthogonalized correctness vector
# --------------------------------------------------------------------------- #

def run_method_A(items, model, tokenizer, output_dir, args,
                 safety_ranked, correct_ranked, head_dim,
                 alpha1, stk, alpha2, ctk):
    fname = make_filename("A", alpha1, stk, alpha2, ctk, 0, 0)

    def mk():
        safety_targets  = select_safe_heads(safety_ranked, stk)
        correct_targets = select_safe_heads(correct_ranked, ctk)
        hooks = []
        for t in safety_targets:
            sv, _, _ = load_head_sv(args.safety_sv_dir, t["layer"], t["head"])
            hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, alpha1))
        for t in correct_targets:
            sv, _, _ = load_head_sv(args.correct_sv_dir_orth, t["layer"], t["head"])
            hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, alpha2))
        info = {
            "method": "A", "alpha1": alpha1, "safety_topk": stk,
            "alpha2": alpha2, "correct_topk": ctk,
        }
        return hooks, info

    run_condition(items, model, tokenizer, mk, output_dir / fname, args)


# --------------------------------------------------------------------------- #
# Method B: joint-head pool (union of top-256 safety + top-256 correctness heads,
#           ranked by min(safety_rank, correct_rank), take top joint_topk)
#           Apply alpha1×safety_SV at safety heads and alpha2×correctness_SV_orth
#           at correctness heads; both accumulate at overlap heads.
# --------------------------------------------------------------------------- #

JOINT_POOL_SIZE = 256  # sentinel rank for heads outside each pool

def build_joint_ranked(safety_ranked: list, correct_ranked: list) -> list:
    """Build joint head list: union of top-JOINT_POOL_SIZE from each ranking.

    Each entry gets safety_rank and correct_rank (1-indexed; JOINT_POOL_SIZE+1
    if not in that pool). Sorted ascending by min(safety_rank, correct_rank).
    """
    safety_map = {}
    for i, h in enumerate(safety_ranked[:JOINT_POOL_SIZE]):
        key = (h["layer"], h["head"])
        safety_map[key] = i + 1  # 1-indexed rank

    correct_map = {}
    for i, h in enumerate(correct_ranked[:JOINT_POOL_SIZE]):
        key = (h["layer"], h["head"])
        correct_map[key] = i + 1

    all_keys = set(safety_map) | set(correct_map)
    sentinel = JOINT_POOL_SIZE + 1
    joint = []
    for key in all_keys:
        sr = safety_map.get(key, sentinel)
        cr = correct_map.get(key, sentinel)
        joint.append({"layer": key[0], "head": key[1],
                      "safety_rank": sr, "correct_rank": cr})

    joint.sort(key=lambda x: (min(x["safety_rank"], x["correct_rank"]),
                               max(x["safety_rank"], x["correct_rank"])))
    return joint


def run_method_B(items, model, tokenizer, output_dir, args,
                 safety_ranked, correct_ranked, head_dim,
                 alpha1, alpha2, jtk):
    fname = make_filename("B", alpha1, 0, alpha2, 0, 0, 0, jtk=jtk)
    joint_ranked = build_joint_ranked(safety_ranked, correct_ranked)
    targets = joint_ranked[:jtk]
    sentinel = JOINT_POOL_SIZE + 1

    def mk():
        hooks = []
        for t in targets:
            if t["safety_rank"] < sentinel:
                sv, _, _ = load_head_sv(args.safety_sv_dir, t["layer"], t["head"])
                hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, alpha1))
            if t["correct_rank"] < sentinel:
                sv, _, _ = load_head_sv(args.correct_sv_dir_orth, t["layer"], t["head"])
                hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, alpha2))
        info = {
            "method": "B", "alpha1": alpha1, "alpha2": alpha2,
            "joint_topk": jtk, "joint_targets": targets,
        }
        return hooks, info

    run_condition(items, model, tokenizer, mk, output_dir / fname, args)


# --------------------------------------------------------------------------- #
# Method C: correctness-only, raw vector (no safety component)
# --------------------------------------------------------------------------- #

def run_method_C(items, model, tokenizer, output_dir, args,
                 correct_ranked, head_dim, alpha2, ctk):
    fname = make_filename("C", 0, 0, alpha2, ctk, 0, 0)

    def mk():
        correct_targets = select_safe_heads(correct_ranked, ctk)
        hooks = []
        for t in correct_targets:
            sv, _, _ = load_head_sv(args.correct_sv_dir_raw, t["layer"], t["head"])
            hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, alpha2))
        info = {"method": "C", "alpha2": alpha2, "correct_topk": ctk}
        return hooks, info

    run_condition(items, model, tokenizer, mk, output_dir / fname, args)


# --------------------------------------------------------------------------- #
# Method D: joint single vector v_j from (safe+correct, vuln+incorrect) pairs
# --------------------------------------------------------------------------- #

def run_method_D(items, model, tokenizer, output_dir, args,
                 safety_ranked, head_dim, alpha, tk):
    """Apply the joint vector v_j at safety-causal top_k heads with single α."""
    fname = make_filename("D", 0, 0, 0, 0, alpha, tk)

    def mk():
        targets = select_safe_heads(safety_ranked, tk)
        hooks = []
        missing = 0
        for t in targets:
            sv_path = Path(args.joint_sv_dir) / f"head_layer_{t['layer']:02d}_head_{t['head']:02d}.pt"
            if not sv_path.exists():
                missing += 1
                continue
            sv = torch.load(sv_path, map_location="cpu")
            hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, alpha))
        if missing:
            print(f"  [D] missing {missing}/{len(targets)} joint SV files")
        info = {"method": "D", "alpha": alpha, "topk": tk}
        return hooks, info

    run_condition(items, model, tokenizer, mk, output_dir / fname, args)


# --------------------------------------------------------------------------- #
# Method E: raw (non-orthogonalized) correctness ablation — same as A but raw
# --------------------------------------------------------------------------- #

def run_method_E(items, model, tokenizer, output_dir, args,
                 safety_ranked, correct_ranked, head_dim,
                 alpha1, stk, alpha2, ctk):
    """
    Ablation: identical to Method A but uses the raw (non-orthogonalized) v_c.
    Hypothesis: since v_c is built from (safe+correct vs safe+incorrect),
    it should not encode the same directional safety signal as v_s.
    If Gram-Schmidt orthogonalization is unnecessary, Method E ≈ Method A.
    """
    fname = make_filename("E", alpha1, stk, alpha2, ctk, 0, 0)

    def mk():
        safety_targets  = select_safe_heads(safety_ranked, stk)
        correct_targets = select_safe_heads(correct_ranked, ctk)
        hooks = []
        for t in safety_targets:
            sv, _, _ = load_head_sv(args.safety_sv_dir, t["layer"], t["head"])
            hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, alpha1))
        for t in correct_targets:
            sv, _, _ = load_head_sv(args.correct_sv_dir_raw, t["layer"], t["head"])
            hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, alpha2))
        info = {
            "method": "E", "alpha1": alpha1, "safety_topk": stk,
            "alpha2": alpha2, "correct_topk": ctk,
            "note": "raw (non-orth) correctness vector",
        }
        return hooks, info

    run_condition(items, model, tokenizer, mk, output_dir / fname, args)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Double steering evaluation (Methods A, C, D, E).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--method", required=True,
                        choices=["A", "B", "C", "D", "E", "all"])
    parser.add_argument("--cwe",        required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--safety_causal_results",  required=True)
    parser.add_argument("--safety_sv_dir",           required=True)
    parser.add_argument("--correct_causal_results",  required=True)
    parser.add_argument("--correct_sv_dir_orth",     required=True,
                        help="Orthogonalized correctness SV (for A)")
    parser.add_argument("--correct_sv_dir_raw",      required=True,
                        help="Raw correctness SV (for C and E)")
    parser.add_argument("--joint_sv_dir",            required=False,
                        help="Joint SV dir (for D); "
                             "e.g. results/double_steering/steering_vectors/{cwe}/joint/head/mean_diff")

    # Unified sweep params (all methods use same α and top_k sets)
    parser.add_argument("--alpha_list", type=float, nargs="+",
                        default=[1.0, 1.5, 2.0, 3.0, 5.0],
                        help="α values for all steering (safety, correctness, joint)")
    parser.add_argument("--topk_list",  type=int,   nargs="+",
                        default=[8, 16, 32, 64, 128],
                        help="Top-k values for all methods")

    # Legacy per-method overrides (if needed for re-running old A/C conditions)
    parser.add_argument("--alpha1_list",       type=float, nargs="+", default=None,
                        help="Override α₁ for safety (A/E); defaults to --alpha_list")
    parser.add_argument("--alpha2_list",       type=float, nargs="+", default=None,
                        help="Override α₂ for correctness (A/C/E); defaults to --alpha_list")
    parser.add_argument("--safety_topk_list",  type=int,   nargs="+", default=None,
                        help="Override safety top-k (A/E); defaults to --topk_list")
    parser.add_argument("--correct_topk_list", type=int,   nargs="+", default=None,
                        help="Override correctness top-k (A/C/E); defaults to --topk_list")
    parser.add_argument("--joint_topk_list",  type=int,   nargs="+", default=None,
                        help="Joint top-k values for Method B; defaults to --topk_list")

    parser.add_argument("--skip_baseline", action="store_true",
                        help="Do not run the unsteered baseline condition")
    parser.add_argument("--skip_safety_only", action="store_true",
                        help="Do not run the safety-only steering conditions")
    parser.add_argument("--only_safety_only", action="store_true", default=False,
                        help="Generate only safety_only/baseline conditions; skip all method-specific loops")

    parser.add_argument("--model",          default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dtype",          default="bfloat16")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample",      action="store_true")
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--top_p",          type=float, default=1.0)

    args = parser.parse_args()

    # Resolve sweep params
    alpha1_list  = args.alpha1_list  or args.alpha_list
    alpha2_list  = args.alpha2_list  or args.alpha_list
    stk_list     = args.safety_topk_list  or args.topk_list
    ctk_list     = args.correct_topk_list or args.topk_list
    jtk_list     = args.joint_topk_list   or args.topk_list

    methods = ["A", "B", "C", "D", "E"] if args.method == "all" else [args.method]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_file) as f:
        items = [json.loads(l) for l in f if l.strip()]
    print(f"Input: {len(items)} prompts from {args.input_file}")

    print(f"\nLoading {args.model!r} ...")
    model, tokenizer = load_model(args.model, args.dtype, args.device)
    _, head_dim = get_head_config(model)
    print(f"  head_dim={head_dim}")

    safety_ranked,  _ = load_head_results(args.safety_causal_results)
    correct_ranked, _ = load_head_results(args.correct_causal_results)
    print(f"  safety ranked heads:  {len(safety_ranked)}")
    print(f"  correct ranked heads: {len(correct_ranked)}")

    if not args.skip_baseline:
        print("\n--- Baseline (no steering) ---")
        run_baseline(items, model, tokenizer, output_dir, args)

    if (not args.skip_safety_only) and ("A" in methods or "B" in methods or "D" in methods or "E" in methods):
        for alpha1 in alpha1_list:
            for stk in stk_list:
                print(f"\n--- Safety-only  α₁={alpha1}  stk={stk} ---")
                run_safety_only(items, model, tokenizer, output_dir, args,
                                safety_ranked, head_dim, alpha1, stk)

    if not args.only_safety_only:
        if "A" in methods:
            for alpha1 in alpha1_list:
                for stk in stk_list:
                    for alpha2 in alpha2_list:
                        for ctk in ctk_list:
                            print(f"\n--- Method A  α₁={alpha1} stk={stk}  α₂={alpha2} ctk={ctk} ---")
                            run_method_A(items, model, tokenizer, output_dir, args,
                                         safety_ranked, correct_ranked, head_dim,
                                         alpha1, stk, alpha2, ctk)

        if "B" in methods:
            for alpha1 in alpha1_list:
                for alpha2 in alpha2_list:
                    for jtk in jtk_list:
                        print(f"\n--- Method B  α₁={alpha1}  α₂={alpha2}  jtk={jtk} ---")
                        run_method_B(items, model, tokenizer, output_dir, args,
                                     safety_ranked, correct_ranked, head_dim,
                                     alpha1, alpha2, jtk)

        if "C" in methods:
            for alpha2 in alpha2_list:
                for ctk in ctk_list:
                    print(f"\n--- Method C  α₂={alpha2}  ctk={ctk} ---")
                    run_method_C(items, model, tokenizer, output_dir, args,
                                 correct_ranked, head_dim, alpha2, ctk)

        if "D" in methods:
            if not args.joint_sv_dir:
                print("ERROR: --joint_sv_dir required for Method D")
            else:
                for alpha in args.alpha_list:
                    for tk in args.topk_list:
                        print(f"\n--- Method D  α={alpha}  tk={tk} ---")
                        run_method_D(items, model, tokenizer, output_dir, args,
                                     safety_ranked, head_dim, alpha, tk)

        if "E" in methods:
            for alpha1 in alpha1_list:
                for stk in stk_list:
                    for alpha2 in alpha2_list:
                        for ctk in ctk_list:
                            print(f"\n--- Method E  α₁={alpha1} stk={stk}  α₂={alpha2} ctk={ctk} ---")
                            run_method_E(items, model, tokenizer, output_dir, args,
                                         safety_ranked, correct_ranked, head_dim,
                                         alpha1, stk, alpha2, ctk)

    print(f"\nAll done. Results in: {output_dir}")


if __name__ == "__main__":
    main()
