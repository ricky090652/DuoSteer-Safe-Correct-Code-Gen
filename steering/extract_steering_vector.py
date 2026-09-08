"""
Steering vector extraction from pre-saved activations.

Activations are read from the output of extract_representations.py:

    {representations_dir}/{model_slug}/{dataset_type}/{cwe_id}/{token_agg}/
        layer_{l:02d}.pt                     → {"safe": Tensor(n, d_model), "vuln": Tensor(n, d_model)}
        head_layer_{l:02d}_head_{h:02d}.pt   → {"safe": Tensor(n, d_head),  "vuln": Tensor(n, d_head)}
        metadata.json

Two methods for computing the steering vector, both sigma-normalised:

  mean_diff  — difference of class centroids (default):
                 v = (mean(safe_acts) − mean(vuln_acts)) / ‖…‖

  probe      — weight vector of a trained linear probe (train_probe.py):
                 v = −w / ‖w‖
               The sign is negated because the probe uses safe=0 / vuln=1,
               so w points toward vuln; −w points toward safe.

Normalisation ensures mean_diff and probe vectors are on the same scale,
making the steering coefficient α directly comparable across methods.

Modes (--mode):
  layer   — compute steering vectors from layer .pt files
  head    — compute steering vectors from head .pt files
  both    — compute both (default)

Methods (--method):
  mean_diff  — mean-difference vectors only
  probe      — probe-weight vectors only (requires --probe_dir)
  both       — both methods (default; requires --probe_dir)

Scope:
  If neither --layer_idx nor --head_idx is provided, all available
  layers / heads are processed and individual .pt files are written.
  Supply --layer_idx (and --head_idx for head mode) to compute a single vector.

Output layout  {output_dir}/
  layer/
    mean_diff/
      steering_vector_layer_{l:02d}.pt  → {"steering_vector", "safe_mean", "vuln_mean",
                                           "raw_norm", "method"}
      metadata.json
    probe/
      steering_vector_layer_{l:02d}.pt  → {"steering_vector", "raw_norm", "method"}
      metadata.json
  head/
    mean_diff/
      steering_vector_head_{l:02d}_{h:02d}.pt
      metadata.json
    probe/
      steering_vector_head_{l:02d}_{h:02d}.pt
      metadata.json

Usage:
  # All layers + heads, both methods
  python extract_steering_vector.py \\
      --representations_dir data/representations \\
      --model  meta-llama/Llama-3.1-8B-Instruct \\
      --cwe_id cwe-022 \\
      --dataset_type safe_only \\
      --token_agg response_mean \\
      --output_dir data/steering_vectors \\
      --probe_dir data/probes

  # Mean-difference only, single layer
  python extract_steering_vector.py \\
      --representations_dir data/representations \\
      --model  meta-llama/Llama-3.1-8B-Instruct \\
      --cwe_id cwe-022 \\
      --dataset_type safe_only \\
      --token_agg response_mean \\
      --output_dir data/steering_vectors \\
      --mode layer --method mean_diff --layer_idx 16

  # Probe only, single head
  python extract_steering_vector.py \\
      --representations_dir data/representations \\
      --model  meta-llama/Llama-3.1-8B-Instruct \\
      --cwe_id cwe-022 \\
      --dataset_type safe_only \\
      --token_agg response_mean \\
      --output_dir data/steering_vectors \\
      --probe_dir data/probes \\
      --mode head --method probe --layer_idx 16 --head_idx 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def slugify(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def find_representations_dir(
    representations_dir: Path,
    model: str,
    dataset_type: str,
    cwe_id: str,
    token_agg: str,
) -> Path:
    model_slug = slugify(model)
    path = representations_dir / model_slug / dataset_type / cwe_id / token_agg
    if not path.exists():
        raise FileNotFoundError(
            f"Representations directory not found: {path}\n"
            "Run extract_representations.py first."
        )
    return path


def find_probe_ckpt_dir(
    probe_dir: Path,
    model: str,
    dataset_type: str,
    cwe_id: str,
    token_agg: str,
    mode: str = "",      # unused; kept for call-site compatibility
) -> Path:
    model_slug = slugify(model)
    path = probe_dir / model_slug / dataset_type / cwe_id / token_agg / "checkpoints"
    if not path.exists():
        raise FileNotFoundError(
            f"Probe checkpoint directory not found: {path}\n"
            "Run train_probe.py first."
        )
    return path


# --------------------------------------------------------------------------- #
# Representation loaders
# --------------------------------------------------------------------------- #

def load_layer_file(rep_dir: Path, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load safe and vuln tensors for layer layer_idx (1-based)."""
    fname = rep_dir / f"layer_{layer_idx:02d}.pt"
    if not fname.exists():
        raise FileNotFoundError(f"Layer file not found: {fname}")
    data = torch.load(fname, map_location="cpu", weights_only=True)
    return data["safe"].float(), data["vuln"].float()


def load_head_file(
    rep_dir: Path, layer_idx: int, head_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load safe and vuln tensors for (layer_idx, head_idx) (both 1-based)."""
    fname = rep_dir / f"head_layer_{layer_idx:02d}_head_{head_idx:02d}.pt"
    if not fname.exists():
        raise FileNotFoundError(f"Head file not found: {fname}")
    data = torch.load(fname, map_location="cpu", weights_only=True)
    return data["safe"].float(), data["vuln"].float()


# --------------------------------------------------------------------------- #
# Probe weight loaders
# --------------------------------------------------------------------------- #

def load_probe_weight_layer(ckpt_dir: Path, layer_idx: int) -> torch.Tensor:
    """
    Load the weight vector from a layer probe checkpoint (1-based layer_idx).
    Returns −w so the vector points toward safe
    (probe labels: safe=0, vuln=1 → w points toward vuln).
    """
    fname = ckpt_dir / f"layer_{layer_idx:02d}_best.pt"
    if not fname.exists():
        raise FileNotFoundError(f"Layer probe checkpoint not found: {fname}")
    state_dict = torch.load(fname, map_location="cpu", weights_only=True)
    w = state_dict["linear.weight"][0].float()   # (dim,)
    return -w   # negate: w points toward vuln, −w points toward safe


def load_probe_weight_head(
    ckpt_dir: Path, layer_idx: int, head_idx: int
) -> torch.Tensor:
    """
    Load the weight vector from a head probe checkpoint (both layer_idx and
    head_idx are 1-based).  Returns −w so the vector points toward safe.
    """
    fname = ckpt_dir / f"head_layer_{layer_idx:02d}_head_{head_idx:02d}_best.pt"
    if not fname.exists():
        raise FileNotFoundError(f"Head probe checkpoint not found: {fname}")
    state_dict = torch.load(fname, map_location="cpu", weights_only=True)
    w = state_dict["linear.weight"][0].float()   # (dim,)
    return -w   # negate: −w points toward safe


# --------------------------------------------------------------------------- #
# Steering vector computation (both methods)
#
# Normalization options (--norm argument):
#
#   sigma  (default)  sv = sv_unit * sigma_proj
#                     where sv_unit   = raw / ||raw||
#                           sigma_proj = std of projections <h_i, sv_unit>
#                                        over the full activation dataset (safe ∪ vuln).
#                     alpha unit: standard deviations along the safe direction.
#                     alpha=1 shifts the projected distribution by exactly 1σ.
#                     Directly comparable across layers, heads, and CWEs.
#
#   raw    (interpretable)  sv = raw  (no normalization)
#                           alpha unit: multiples of the centroid-separation distance.
#                           alpha=1 moves exactly one centroid-separation distance.
#                           Simple and interpretable; not comparable in absolute
#                           activation-magnitude across spaces.
#
# Both store raw_norm and proj_sigma in the .pt file for reference.
# --------------------------------------------------------------------------- #

def _proj_sigma(safe: torch.Tensor, vuln: torch.Tensor, sv_unit: torch.Tensor) -> float:
    """
    Standard deviation of scalar projections <h, sv_unit> over safe ∪ vuln activations.
    Used for sigma normalization so alpha=1 ≡ 1σ shift in the safe direction.
    """
    all_acts = torch.cat([safe, vuln], dim=0).float()   # (N, dim)
    projections = all_acts @ sv_unit.float()             # (N,)
    return projections.std().item()


def mean_diff_steering_vector(
    safe: torch.Tensor,
    vuln: torch.Tensor,
    norm: str = "sigma",
) -> dict:
    """
    Compute mean-difference steering vector with the requested normalization.

    Parameters
    ----------
    safe, vuln : Tensor  (n_pairs, dim)  float activations
    norm       : "sigma" | "raw"

    Returns a dict always containing:
      steering_vector  — the normalized vector used for steering
      raw_norm         — ||safe_mean - vuln_mean||
      proj_sigma       — std of projections onto the unit direction
      norm             — normalization method used
      safe_mean, vuln_mean
    """
    safe_mean = safe.mean(dim=0)
    vuln_mean = vuln.mean(dim=0)
    raw       = safe_mean - vuln_mean
    raw_norm  = raw.norm().item()
    sv_unit   = raw / raw_norm

    sigma = _proj_sigma(safe, vuln, sv_unit)

    if norm == "sigma":
        # sv = sv_unit * sigma so that h' = h + alpha*sv shifts projection by alpha*sigma = alpha σ
        # alpha=1 ≡ exactly 1σ shift toward safe; comparable across layers, heads, CWEs
        sv = sv_unit * sigma
    elif norm == "raw":
        sv = raw              # alpha=1 ↔ 1 centroid-sep distance
    else:
        raise ValueError(f"Unknown norm: {norm!r}  (choices: sigma, raw)")

    return {
        "steering_vector": sv,
        "sv_unit":          sv_unit,   # always stored for reference
        "safe_mean":        safe_mean,
        "vuln_mean":        vuln_mean,
        "raw_norm":         raw_norm,
        "proj_sigma":       sigma,
        "norm":             norm,
        "method":           "mean_diff",
    }


def probe_steering_vector(
    w_toward_safe: torch.Tensor,
    safe: torch.Tensor | None = None,
    vuln: torch.Tensor | None = None,
    norm: str = "sigma",
) -> dict:
    """
    Compute probe-weight steering vector with the requested normalization.

    w_toward_safe : (dim,)  probe weight negated to point toward safe
    safe, vuln    : activations needed when norm="sigma"; ignored for "raw"
    norm          : "sigma" | "raw"
    """
    raw_norm = w_toward_safe.norm().item()
    sv_unit  = w_toward_safe / raw_norm

    if norm == "sigma":
        if safe is None or vuln is None:
            raise ValueError("safe and vuln activations required for sigma normalization")
        sigma = _proj_sigma(safe, vuln, sv_unit)
        sv = sv_unit * sigma
    elif norm == "raw":
        sigma = float("nan") if (safe is None or vuln is None) else _proj_sigma(safe, vuln, sv_unit)
        sv = w_toward_safe     # unnormalised probe weight
    else:
        raise ValueError(f"Unknown norm: {norm!r}  (choices: sigma, raw)")

    return {
        "steering_vector": sv,
        "sv_unit":          sv_unit,
        "raw_norm":         raw_norm,
        "proj_sigma":       sigma,
        "norm":             norm,
        "method":           "probe",
    }


# --------------------------------------------------------------------------- #
# Discovery helpers
# --------------------------------------------------------------------------- #

def discover_layers(rep_dir: Path) -> list[int]:
    """Return sorted 1-based layer indices from layer_XX.pt files."""
    indices = []
    for f in rep_dir.glob("layer_*.pt"):
        m = re.match(r"layer_(\d+)\.pt", f.name)
        if m:
            indices.append(int(m.group(1)))   # already 1-based in filename
    return sorted(indices)


def discover_heads(rep_dir: Path) -> list[tuple[int, int]]:
    """Return sorted (layer_idx, head_idx) pairs (both 1-based) from head files."""
    pairs = []
    for f in rep_dir.glob("head_layer_*_head_*.pt"):
        m = re.match(r"head_layer_(\d+)_head_(\d+)\.pt", f.name)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))  # both 1-based
    return sorted(pairs)


# --------------------------------------------------------------------------- #
# Extraction routines
# --------------------------------------------------------------------------- #

def extract_layers(
    rep_dir: Path,
    output_dir: Path,
    layer_indices: list[int],
    meta_base: dict,
    methods: list[str],
    probe_ckpt_dir: Path | None,
    norm: str = "sigma",
) -> None:
    for method in methods:
        out_dir = output_dir / "layer" / method
        out_dir.mkdir(parents=True, exist_ok=True)

        results_meta = []
        for l in layer_indices:
            fname = f"steering_vector_layer_{l:02d}.pt"

            if method == "mean_diff":
                safe, vuln = load_layer_file(rep_dir, l)
                vec_data = mean_diff_steering_vector(safe, vuln, norm=norm)
                n_pairs = safe.shape[0]
            else:  # probe
                try:
                    w = load_probe_weight_head_or_layer(probe_ckpt_dir, l, None)
                except FileNotFoundError as e:
                    print(f"  [probe] layer {l:2d}: skipped — {e}")
                    continue
                safe, vuln = load_layer_file(rep_dir, l) if norm == "sigma" else (None, None)
                vec_data = probe_steering_vector(w, safe=safe, vuln=vuln, norm=norm)
                n_pairs = meta_base.get("n_pairs")

            torch.save(vec_data, out_dir / fname)

            print(f"  [{method}] layer {l:2d}: dim={vec_data['steering_vector'].shape[0]:5d}  "
                  f"raw_norm={vec_data['raw_norm']:.4f}  "
                  f"proj_sigma={vec_data['proj_sigma']:.4f}  "
                  f"norm={norm}  n_pairs={n_pairs}")

            results_meta.append({
                "layer_idx":  l,
                "file":       fname,
                "dim":        vec_data["steering_vector"].shape[0],
                "n_pairs":    n_pairs,
                "raw_norm":   vec_data["raw_norm"],
                "proj_sigma": vec_data["proj_sigma"],
                "norm":       norm,
            })

        meta = {**meta_base, "mode": "layer", "method": method, "norm": norm,
                "layers": results_meta}
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  → {len(results_meta)} layer vectors [{method}, norm={norm}] saved to {out_dir}")


def extract_heads(
    rep_dir: Path,
    output_dir: Path,
    head_indices: list[tuple[int, int]],
    meta_base: dict,
    methods: list[str],
    probe_ckpt_dir: Path | None,
    norm: str = "sigma",
) -> None:
    for method in methods:
        out_dir = output_dir / "head" / method
        out_dir.mkdir(parents=True, exist_ok=True)

        results_meta = []
        for l, h in head_indices:
            fname = f"steering_vector_head_{l:02d}_{h:02d}.pt"

            if method == "mean_diff":
                safe, vuln = load_head_file(rep_dir, l, h)
                vec_data = mean_diff_steering_vector(safe, vuln, norm=norm)
                n_pairs = safe.shape[0]
            else:  # probe
                try:
                    w = load_probe_weight_head(probe_ckpt_dir, l, h)
                except FileNotFoundError:
                    continue
                safe, vuln = load_head_file(rep_dir, l, h) if norm == "sigma" else (None, None)
                vec_data = probe_steering_vector(w, safe=safe, vuln=vuln, norm=norm)
                n_pairs = meta_base.get("n_pairs")

            torch.save(vec_data, out_dir / fname)

            results_meta.append({
                "layer_idx":  l,
                "head_idx":   h,
                "file":       fname,
                "dim":        vec_data["steering_vector"].shape[0],
                "n_pairs":    n_pairs,
                "raw_norm":   vec_data["raw_norm"],
                "proj_sigma": vec_data["proj_sigma"],
                "norm":       norm,
            })

        meta = {**meta_base, "mode": "head", "method": method, "norm": norm,
                "heads": results_meta}
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  → {len(results_meta)} head vectors [{method}, norm={norm}] saved to {out_dir}")


def load_probe_weight_head_or_layer(
    ckpt_dir: Path, layer_idx: int, head_idx: int | None
) -> torch.Tensor:
    """Dispatch to layer or head probe loader based on head_idx."""
    if head_idx is None:
        return load_probe_weight_layer(ckpt_dir, layer_idx)
    return load_probe_weight_head(ckpt_dir, layer_idx, head_idx)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    rep_dir = find_representations_dir(
        Path(args.representations_dir),
        args.model,
        args.dataset_type,
        args.cwe_id,
        args.token_agg,
    )
    print(f"Representations directory: {rep_dir}")

    # Resolve which methods to run
    methods = ["mean_diff", "probe"] if args.method == "both" else [args.method]

    # Validate probe_dir if probe method is requested
    probe_layer_ckpt_dir = None
    probe_head_ckpt_dir  = None
    if "probe" in methods:
        if not args.probe_dir:
            raise ValueError("--probe_dir is required when --method includes 'probe'")
        probe_base = Path(args.probe_dir)
        do_layer = args.mode in ("layer", "both")
        do_head  = args.mode in ("head",  "both")
        if do_layer:
            probe_layer_ckpt_dir = find_probe_ckpt_dir(
                probe_base, args.model, args.dataset_type, args.cwe_id, args.token_agg, "layer"
            )
            print(f"Layer probe checkpoints:   {probe_layer_ckpt_dir}")
        if do_head:
            probe_head_ckpt_dir = find_probe_ckpt_dir(
                probe_base, args.model, args.dataset_type, args.cwe_id, args.token_agg, "head"
            )
            print(f"Head probe checkpoints:    {probe_head_ckpt_dir}")

    # Load run metadata if present
    run_meta_path = rep_dir / "metadata.json"
    run_meta = {}
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)

    output_dir = (
        Path(args.output_dir)
        / slugify(args.model)
        / args.dataset_type
        / args.cwe_id
        / args.token_agg
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory:          {output_dir}")

    meta_base = {
        "model":           args.model,
        "cwe_id":          args.cwe_id,
        "dataset_type":    args.dataset_type,
        "token_agg":       args.token_agg,
        "representations_dir": str(rep_dir),
        "n_pairs":         run_meta.get("n_pairs"),
        "n_layers":        run_meta.get("n_layers"),
        "n_heads":         run_meta.get("n_heads"),
        "head_dim":        run_meta.get("head_dim"),
        "hidden_dim":      run_meta.get("hidden_dim"),
    }

    do_layer = args.mode in ("layer", "both")
    do_head  = args.mode in ("head",  "both")

    # ---- Layer steering vectors ----
    if do_layer:
        if args.layer_idx is not None:
            layer_indices = [args.layer_idx]
        else:
            layer_indices = discover_layers(rep_dir)
            if not layer_indices:
                print("No layer_XX.pt files found — skipping layer mode.")
                do_layer = False

        if do_layer:
            print(f"\nExtracting layer steering vectors "
                  f"(layers {layer_indices}, methods={methods}) ...")
            extract_layers(
                rep_dir, output_dir, layer_indices, meta_base,
                methods, probe_layer_ckpt_dir, norm=args.norm,
            )

    # ---- Head steering vectors ----
    if do_head:
        if args.layer_idx is not None and args.head_idx is not None:
            head_indices = [(args.layer_idx, args.head_idx)]
        elif args.layer_idx is not None:
            all_heads = discover_heads(rep_dir)
            head_indices = [(l, h) for l, h in all_heads if l == args.layer_idx]
        else:
            head_indices = discover_heads(rep_dir)
            if not head_indices:
                print("No head_layer_*_head_*.pt files found — skipping head mode.")
                do_head = False

        if do_head:
            print(f"\nExtracting head steering vectors ({len(head_indices)} heads, "
                  f"methods={methods}) ...")
            extract_heads(
                rep_dir, output_dir, head_indices, meta_base,
                methods, probe_head_ckpt_dir, norm=args.norm,
            )

    print("\nDone.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute steering vectors from pre-saved activations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--representations_dir", default="data/representations",
        help="Root directory written by extract_representations.py (default: data/representations)",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name used in extract_representations.py",
    )
    parser.add_argument(
        "--cwe_id", required=True,
        help="CWE to process, e.g. cwe-022",
    )
    parser.add_argument(
        "--dataset_type", default="safe_only",
        help="Dataset type: safe_only | cross_group (default: safe_only)",
    )
    parser.add_argument(
        "--token_agg", default="response_mean",
        choices=["response_last", "response_mean"],
        help="Aggregation used when saving representations (default: response_mean)",
    )
    parser.add_argument(
        "--output_dir", default="data/steering_vectors",
        help="Root output directory (default: data/steering_vectors)",
    )
    parser.add_argument(
        "--probe_dir", default="data/probes",
        help="Root directory written by train_probe.py (default: data/probes).\n"
             "Required when --method includes 'probe'.",
    )
    parser.add_argument(
        "--mode", default="both", choices=["layer", "head", "both"],
        help="Which representations to use (default: both)",
    )
    parser.add_argument(
        "--method", default="both", choices=["mean_diff", "probe", "both"],
        help="Steering vector method (default: both):\n"
             "  mean_diff — normalised mean(safe) − mean(vuln)\n"
             "  probe     — normalised probe weight −w (negated so it points toward safe)\n"
             "  both      — compute both methods",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=None,
        help="1-based layer index. If omitted, process all available layers.",
    )
    parser.add_argument(
        "--head_idx", type=int, default=None,
        help="1-based head index (requires --layer_idx). If omitted, process all heads.",
    )
    parser.add_argument(
        "--norm", default="sigma", choices=["sigma", "raw"],
        help="Steering vector normalization (default: sigma):\n"
             "  sigma — sv = sv_unit * proj_sigma; alpha=1 ≡ 1σ shift in projection space\n"
             "  raw   — sv = raw (unnormalized; alpha=1 ≡ 1 centroid-separation distance)",
    )
    main(parser.parse_args())
