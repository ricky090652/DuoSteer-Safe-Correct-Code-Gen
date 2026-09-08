"""
DS-8d: Orthogonalize correctness steering vectors against safety steering vectors.

For each (layer, head), removes the component of the correctness vector that is
parallel to the safety vector so that applying the correctness offset at inference
cannot partially re-introduce the vulnerable features that the safety offset suppressed.

Formula (Gram-Schmidt):
    v_orth = v_correct - (v_correct · v_safe / |v_safe|²) * v_safe

Usage:
    python orthogonalize_vectors.py --dataset_type llama31-8b_intra
    python orthogonalize_vectors.py --dataset_type llama31-8b_intra --verify_only  # cosine sims only
    python orthogonalize_vectors.py --dataset_type llama31-8b_intra --cwe cwe-022 \\
        --model meta-llama/Meta-Llama-3.1-8B-Instruct --safety_sv_base data/steering_vectors
"""

import argparse
import json
import re
from pathlib import Path

import torch

CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]

# Populated from CLI args in main().
MODEL_SLUG = ""
DATASET_TYPE = ""
SAFETY_SV_BASE  = Path("data/steering_vectors")
CORRECT_SV_BASE = Path("results/double_steering/steering_vectors")
CORRECT_TOKEN_AGG = "response_last"


def slugify(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def safety_sv_path(cwe: str, mode: str, layer: int, head: int = None) -> Path:
    # Matches extract_steering_vector.py: {base}/{model_slug}/{dataset_type}/{cwe}/response_mean/...
    base = SAFETY_SV_BASE / MODEL_SLUG / DATASET_TYPE / cwe / "response_mean" / mode / "mean_diff"
    if mode == "layer":
        return base / f"steering_vector_layer_{layer:02d}.pt"
    else:
        return base / f"steering_vector_head_{layer:02d}_{head:02d}.pt"


def correct_sv_raw_base(cwe: str, mode: str) -> Path:
    # extract_steering_vector.py appends model_slug/dataset_type/cwe_id/token_agg
    return (CORRECT_SV_BASE / cwe / "raw"
            / MODEL_SLUG / f"{cwe}_train" / cwe / CORRECT_TOKEN_AGG
            / mode / "mean_diff")


def correct_sv_path(cwe: str, mode: str, layer: int, head: int = None) -> Path:
    base = correct_sv_raw_base(cwe, mode)
    if mode == "layer":
        return base / f"steering_vector_layer_{layer:02d}.pt"
    else:
        return base / f"steering_vector_head_{layer:02d}_{head:02d}.pt"


def orth_sv_path(cwe: str, mode: str, layer: int, head: int = None) -> Path:
    base = CORRECT_SV_BASE / cwe / "orthogonal" / mode
    if mode == "layer":
        return base / f"steering_vector_layer_{layer:02d}.pt"
    else:
        return base / f"steering_vector_head_{layer:02d}_{head:02d}.pt"


def orthogonalize(v_correct: torch.Tensor, v_safe: torch.Tensor) -> tuple:
    """Gram-Schmidt: remove projection of v_correct onto v_safe."""
    v_safe_f   = v_safe.float()
    v_correct_f = v_correct.float()
    safe_norm_sq = (v_safe_f * v_safe_f).sum()
    if safe_norm_sq < 1e-12:
        return v_correct_f, 0.0, 0.0

    projection = ((v_correct_f * v_safe_f).sum() / safe_norm_sq) * v_safe_f
    v_orth = v_correct_f - projection

    cos_sim = float(
        (v_correct_f * v_safe_f).sum()
        / (v_correct_f.norm() * v_safe_f.norm() + 1e-12)
    )
    proj_fraction = float(projection.norm() / (v_correct_f.norm() + 1e-12))
    return v_orth, cos_sim, proj_fraction


def process_cwe(cwe: str, verify_only: bool = False) -> dict:
    stats = {}

    for mode in ("layer", "head"):
        # Enumerate available correctness vectors
        raw_base = correct_sv_raw_base(cwe, mode)
        if not raw_base.exists():
            print(f"  {cwe} {mode}: raw directory not found, skipping")
            continue

        files = sorted(raw_base.glob("steering_vector_*.pt"))
        if not files:
            print(f"  {cwe} {mode}: no vector files found")
            continue

        orth_base = CORRECT_SV_BASE / cwe / "orthogonal" / mode
        if not verify_only:
            orth_base.mkdir(parents=True, exist_ok=True)

        for sv_file in files:
            name = sv_file.stem  # e.g. steering_vector_layer_09 or steering_vector_head_09_03

        n_processed = n_zero_orth = 0
        cos_sims = []

        for sv_file in files:
            name = sv_file.stem
            parts = name.split("_")

            if mode == "layer":
                layer = int(parts[-1])
                head  = None
            else:
                layer = int(parts[-2])
                head  = int(parts[-1])

            # Load safety vector
            s_path = safety_sv_path(cwe, mode, layer, head)
            if not s_path.exists():
                continue

            c_data = torch.load(sv_file, map_location="cpu", weights_only=True)
            s_data = torch.load(s_path,  map_location="cpu", weights_only=True)
            v_correct = c_data["steering_vector"].float()
            v_safe    = s_data["steering_vector"].float()

            v_orth, cos_sim, proj_frac = orthogonalize(v_correct, v_safe)

            # Residual cosine similarity after orthogonalization (should be ~0)
            v_safe_f2 = v_safe.float()
            residual_cos = float(
                (v_orth * v_safe_f2).sum()
                / (v_orth.norm() * v_safe_f2.norm() + 1e-12)
            )
            key = name.replace("steering_vector_", "")
            stats[f"{mode}/{key}"] = {
                "raw_norm":                  float(v_correct.norm()),
                "orth_norm":                 float(v_orth.norm()),
                "cosine_sim_with_safety":    round(cos_sim, 6),       # pre-orth overlap
                "residual_cosine_sim":       round(residual_cos, 8),  # should be ~0
                "projection_removed_frac":   round(proj_frac, 6),
            }
            cos_sims.append(abs(residual_cos))

            if float(v_orth.norm()) < 1e-8:
                n_zero_orth += 1

            if not verify_only:
                out_path = orth_sv_path(cwe, mode, layer, head)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_dict = dict(c_data)
                save_dict["steering_vector"] = v_orth.to(c_data["steering_vector"].dtype)
                save_dict["orthogonalized"]  = True
                torch.save(save_dict, out_path)

            n_processed += 1

        avg_pre_cos = sum(abs(stats[f"{mode}/{sv_file.stem.replace('steering_vector_', '')}"]["cosine_sim_with_safety"]) for sv_file in files if f"{mode}/{sv_file.stem.replace('steering_vector_', '')}" in stats) / max(n_processed, 1)
        avg_res_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
        print(f"  {cwe} {mode}: {n_processed} vectors  "
              f"avg_pre_cos={avg_pre_cos:.4f}  avg_residual_cos={avg_res_cos:.2e}  "
              f"zero_orth={n_zero_orth}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwe", nargs="+", default=CWES)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model id (slugified into the vector paths)")
    parser.add_argument("--dataset_type", required=True,
                        help="Dataset type used when the safety vectors were extracted "
                             "(the {dataset_type} path component from extract_steering_vector.py)")
    parser.add_argument("--safety_sv_base", default="data/steering_vectors",
                        help="Safety steering-vector root (default: data/steering_vectors)")
    parser.add_argument("--correct_sv_base", default=None,
                        help="Override the correctness steering-vector base directory")
    parser.add_argument("--verify_only", action="store_true",
                        help="Check cosine similarities only; do not write files")
    args = parser.parse_args()
    global SAFETY_SV_BASE, CORRECT_SV_BASE, MODEL_SLUG, DATASET_TYPE
    MODEL_SLUG = slugify(args.model)
    DATASET_TYPE = args.dataset_type
    SAFETY_SV_BASE = Path(args.safety_sv_base)
    if args.correct_sv_base:
        CORRECT_SV_BASE = Path(args.correct_sv_base)

    all_stats = {}
    for cwe in args.cwe:
        print(f"\n=== {cwe} ===")
        stats = process_cwe(cwe, verify_only=args.verify_only)
        all_stats.update({f"{cwe}/{k}": v for k, v in stats.items()})

        # Write stats per CWE
        if not args.verify_only:
            stats_path = CORRECT_SV_BASE / cwe / "orthogonalization_stats.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  stats written to {stats_path}")

    # Post-verification: residual cosine similarity after orth should be < 0.01
    max_residual = max(
        (abs(v["residual_cosine_sim"]) for v in all_stats.values()), default=0.0
    )
    print(f"\nVerification: max |residual_cosine_sim| = {max_residual:.2e}  (should be <0.01)")
    if max_residual >= 0.01:
        print("  WARNING: residual cosine too high — check orthogonalization")
    else:
        print("  PASS")


if __name__ == "__main__":
    main()
