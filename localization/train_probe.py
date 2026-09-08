"""
Train a linear probe on extracted LLM representations.

For each layer (or each attention head per layer), trains a binary linear
classifier to predict whether the code is vulnerable (label=1) or safe (label=0).

The train/val split is performed at the question level: all pairs from the same
question are assigned to the same split, preventing any question leakage.

Modes:
  layer  — one probe per transformer layer   (n_layers+1 total)
  head   — one probe per (layer, head)        (n_layers × n_heads total)

Outputs saved to {output_dir}/:
  checkpoints/
    layer_01_best.pt                   (layer mode)
    head_layer_01_head_01_best.pt      (head mode)
    ...
  plots/
    [layer mode]
      val_curves.png          — val accuracy over epochs, one line per layer
      best_per_layer.png      — best val accuracy across all layers
    [head mode]
      head_heatmap.png        — heatmap of best val accuracy (layers × heads)

Usage:
  # Layer-level probes
  python train_probe.py \\
      --rep_dir data/representations/.../response_last \\
      --mode layer

  # Attention-head probes
  python train_probe.py \\
      --rep_dir data/representations/.../response_last \\
      --mode head \\
      --epochs 50
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


# --------------------------------------------------------------------------- #
# Linear probe model
# --------------------------------------------------------------------------- #

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)   # (batch,) logits


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

def load_metadata(rep_dir: Path) -> dict:
    path = rep_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found in {rep_dir}")
    with open(path) as f:
        return json.load(f)


def make_split_indices(
    metadata: dict,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Group row indices by question (src_id), shuffle at question level, split.
    Guarantees no question appears in both train and val.

    Uses src_id (question-level key) when available in metadata; falls back to
    the pair id. Augmented datasets mix safe_only and cross_group pairs from the
    same question — src_id ensures both go to the same split.
    """
    question_to_indices: defaultdict[str, list[int]] = defaultdict(list)
    for pair in metadata["pairs"]:
        key = pair.get("src_id") or pair["id"]
        question_to_indices[key].append(pair["index"])

    question_ids = list(question_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(question_ids)

    n_val = max(1, round(len(question_ids) * val_ratio))
    val_qids   = set(question_ids[:n_val])
    train_qids = set(question_ids[n_val:])

    train_idx = sorted(i for qid in train_qids for i in question_to_indices[qid])
    val_idx   = sorted(i for qid in val_qids   for i in question_to_indices[qid])
    return train_idx, val_idx


def build_tensors(
    pt_data: dict,
    indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    From {"safe": Tensor(n_pairs, dim), "vuln": Tensor(n_pairs, dim)},
    build X (2*|indices|, dim) and y (2*|indices|,) in float32.
    Labels: safe=0, vuln=1.
    """
    idx = torch.tensor(indices, dtype=torch.long)
    safe = pt_data["safe"][idx].float()
    vuln = pt_data["vuln"][idx].float()
    X = torch.cat([safe, vuln], dim=0).to(device)
    y = torch.cat([
        torch.zeros(len(idx), dtype=torch.float32),
        torch.ones( len(idx), dtype=torch.float32),
    ], dim=0).to(device)
    return X, y


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_one_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val:   torch.Tensor,
    y_val:   torch.Tensor,
    epochs:  int,
    lr:      float,
    batch_size: int,
    device:  torch.device,
    ckpt_path: Path,
) -> tuple[list[float], float]:
    """
    Train a single LinearProbe and return (val_acc_per_epoch, best_val_acc).
    Saves the best checkpoint to ckpt_path.
    """
    dim   = X_train.shape[1]
    probe = LinearProbe(dim).to(device)
    optimizer = Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    val_accs: list[float] = []
    best_acc  = 0.0

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_train), device=device)
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        for start in range(0, len(X_train), batch_size):
            xb = X_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]
            optimizer.zero_grad()
            criterion(probe(xb), yb).backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            logits = probe(X_val)
            acc = ((logits > 0) == y_val.bool()).float().mean().item()

        val_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(probe.state_dict(), ckpt_path)

    return val_accs, best_acc


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #

def plot_val_curves(
    all_curves: dict[str, list[float]],   # label → val_acc per epoch
    out_path: Path,
    title: str = "Validation accuracy per epoch",
) -> None:
    """One line per layer / head, coloured by a sequential colormap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(all_curves)
    cmap = plt.cm.get_cmap("plasma", n)

    for i, (label, curve) in enumerate(all_curves.items()):
        ax.plot(curve, color=cmap(i), linewidth=0.9, alpha=0.8, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # Only show legend when few enough lines
    if n <= 20:
        ax.legend(fontsize=7, ncol=2, loc="lower right")
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Layer index")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_best_per_layer(
    best_accs: list[float],
    layer_labels: list[str],
    out_path: Path,
) -> None:
    """Bar chart of best val accuracy per layer, coloured by accuracy (Blues: dark=high)."""
    fig, ax = plt.subplots(figsize=(max(8, len(best_accs) * 0.35), 5))
    x = np.arange(len(best_accs))

    vmin = max(0.0, min(best_accs) - 0.05)
    vmax = min(1.0, max(best_accs) + 0.05)
    norm  = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap  = plt.cm.get_cmap("Blues")
    colors = [cmap(norm(a)) for a in best_accs]

    ax.bar(x, best_accs, color=colors, edgecolor="none")
    ax.plot(x, best_accs, "o-", color="#333333", linewidth=1.2,
            markersize=3, alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Chance (0.5)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Best Validation Accuracy")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Probe accuracy across transformer layers")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=7,
                       rotation=45 if len(x) > 16 else 0)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_head_heatmap(
    acc_matrix: np.ndarray,   # (n_layers, n_heads)  raw accuracy, row 0 = layer 1
    out_path: Path,
) -> None:
    """
    Heatmap of best probe accuracy.
      y-axis — layer number, layer 1 at bottom, highest layer at top
      x-axis — head rank within that layer (0 = highest accuracy, sorted independently per layer)

    Blues colormap: dark = high accuracy, light = low.
    No per-cell annotations since heads are already sorted by rank.
    """
    n_layers, n_heads = acc_matrix.shape   # n_layers rows = layers 1..n_layers

    # For each layer (row), sort heads by accuracy descending
    sorted_acc = np.zeros_like(acc_matrix)
    for l in range(n_layers):
        order = np.argsort(acc_matrix[l])[::-1]
        sorted_acc[l] = acc_matrix[l, order]

    # Flip vertically so layer 1 is at the bottom
    display = sorted_acc[::-1]

    vmin = max(0.4, sorted_acc.min() - 0.02)
    vmax = min(1.0, sorted_acc.max() + 0.02)

    fig_w = max(10, n_heads  * 0.45)
    fig_h = max(6,  n_layers * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        display,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Best Validation Accuracy", fontsize=10)

    ax.set_xlabel("Head rank within layer  (0 = best)", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(
        "Probe accuracy per layer × head\n"
        "(heads sorted high → low within each layer)",
        fontsize=12,
    )

    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([str(r) for r in range(n_heads)], fontsize=7)

    # y-ticks: display row 0 = highest layer, display row n_layers-1 = layer 1
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([str(n_layers - l) for l in range(n_layers)], fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# --------------------------------------------------------------------------- #
# Layer-mode training
# --------------------------------------------------------------------------- #

def run_layer_mode(
    rep_dir:    Path,
    train_idx:  list[int],
    val_idx:    list[int],
    metadata:   dict,
    epochs:     int,
    lr:         float,
    batch_size: int,
    device:     torch.device,
    ckpt_dir:   Path,
    plot_dir:   Path,
) -> None:
    n_layers = metadata["n_layers"]
    layer_files = sorted(rep_dir.glob("layer_*.pt"))

    if not layer_files:
        print("No layer_*.pt files found — skipping layer mode.")
        return

    all_curves: dict[str, list[float]] = {}
    best_accs:  list[float] = []

    print(f"Training layer probes  ({len(layer_files)} layers, {epochs} epochs each) ...")

    for pt_file in layer_files:
        layer_idx = int(pt_file.stem.split("_")[1])
        label = f"layer {layer_idx:02d}"

        data = torch.load(pt_file, map_location="cpu")
        X_train, y_train = build_tensors(data, train_idx, device)
        X_val,   y_val   = build_tensors(data, val_idx,   device)
        del data

        ckpt_path = ckpt_dir / f"layer_{layer_idx:02d}_best.pt"

        curves, best = train_one_probe(
            X_train, y_train, X_val, y_val,
            epochs, lr, batch_size, device, ckpt_path,
        )

        all_curves[label] = curves
        best_accs.append((layer_idx, best))

        print(f"  layer {layer_idx:02d}  best_val_acc={best:.4f}  "
              f"train={len(train_idx)*2}  val={len(val_idx)*2}")

    # --- metadata JSON ---
    # List of all layers sorted globally by accuracy descending.
    results = sorted(
        [{"layer": l, "val_accuracy": round(acc, 6)} for l, acc in best_accs],
        key=lambda x: x["val_accuracy"],
        reverse=True,
    )
    meta_path = plot_dir / "layer_accuracy_results.json"
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {meta_path}")
    print(f"  Best layer: {results[0]}")

    # Plot 1: val accuracy curves for all layers
    accs_in_order = [acc for _, acc in best_accs]
    plot_val_curves(
        all_curves,
        plot_dir / "val_curves.png",
        title="Validation accuracy over epochs (layer probes)",
    )

    # Plot 2: best accuracy per layer
    layer_labels = [str(l) for l, _ in best_accs]
    plot_best_per_layer(accs_in_order, layer_labels, plot_dir / "best_per_layer.png")


# --------------------------------------------------------------------------- #
# Head-mode training
# --------------------------------------------------------------------------- #

def run_head_mode(
    rep_dir:    Path,
    train_idx:  list[int],
    val_idx:    list[int],
    metadata:   dict,
    epochs:     int,
    lr:         float,
    batch_size: int,
    device:     torch.device,
    ckpt_dir:   Path,
    plot_dir:   Path,
) -> None:
    n_layers = metadata["n_layers"]
    n_heads  = metadata["n_heads"]

    acc_matrix = np.full((n_layers, n_heads), np.nan)
    total = n_layers * n_heads

    print(f"Training head probes  ({total} probes: {n_layers} layers × {n_heads} heads, "
          f"{epochs} epochs each) ...")

    done = 0
    for l in range(n_layers):
        for h in range(n_heads):
            pt_file = rep_dir / f"head_layer_{l+1:02d}_head_{h+1:02d}.pt"
            if not pt_file.exists():
                done += 1
                continue

            data = torch.load(pt_file, map_location="cpu")
            X_train, y_train = build_tensors(data, train_idx, device)
            X_val,   y_val   = build_tensors(data, val_idx,   device)
            del data

            ckpt_path = ckpt_dir / f"head_layer_{l+1:02d}_head_{h+1:02d}_best.pt"

            _, best = train_one_probe(
                X_train, y_train, X_val, y_val,
                epochs, lr, batch_size, device, ckpt_path,
            )

            acc_matrix[l, h] = best
            done += 1
            print(f"  [{done}/{total}] layer {l+1:02d} head {h+1:02d}  "
                  f"best_val_acc={best:.4f}", end="\r")

    print()  # newline after \r

    # Replace NaN (missing files) with 0
    acc_matrix = np.nan_to_num(acc_matrix, nan=0.0)

    # --- metadata JSON ---
    # Flat list of all (layer, head) results, sorted globally by accuracy descending.
    # Fields: layer, head, val_accuracy, rank_in_layer
    # rank_in_layer=0 means that head had the highest accuracy in its layer.
    results = []
    for l in range(n_layers):
        layer_order = np.argsort(acc_matrix[l])[::-1]   # best first
        rank_map = {int(h): int(r) for r, h in enumerate(layer_order)}
        for h in range(n_heads):
            results.append({
                "layer":         l + 1,
                "head":          h + 1,
                "val_accuracy":  round(float(acc_matrix[l, h]), 6),
                "rank_in_layer": rank_map[h],
            })
    results.sort(key=lambda x: x["val_accuracy"], reverse=True)

    meta_path = plot_dir / "head_accuracy_results.json"
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {meta_path}")

    # --- heatmap ---
    plot_head_heatmap(acc_matrix, plot_dir / "head_heatmap.png")

    # Summary statistics
    best = results[0]
    print(f"\nBest head overall:  layer={best['layer']}  head={best['head']}  "
          f"acc={best['val_accuracy']:.4f}")
    print(f"Mean head accuracy: {acc_matrix.mean():.4f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    rep_dir = Path(args.rep_dir)
    metadata = load_metadata(rep_dir)

    print(f"Representation dir : {rep_dir}")
    print(f"Model              : {metadata.get('model', '?')}")
    print(f"CWE                : {metadata.get('cwe_id', '?')}")
    print(f"Total pairs        : {metadata['n_pairs']}")
    print(f"Mode               : {args.mode}")

    train_idx, val_idx = make_split_indices(metadata, args.val_ratio, args.seed)

    # Verify no leakage
    train_qids = {p["id"] for p in metadata["pairs"] if p["index"] in set(train_idx)}
    val_qids   = {p["id"] for p in metadata["pairs"] if p["index"] in set(val_idx)}
    assert len(train_qids & val_qids) == 0, "Question ID leakage detected!"

    print(f"Train              : {len(train_idx)} pairs  ({len(train_qids)} questions)")
    print(f"Val                : {len(val_idx)} pairs  ({len(val_qids)} questions)")

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    plot_dir = out_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device             : {device}\n")

    if args.mode == "layer":
        run_layer_mode(
            rep_dir, train_idx, val_idx, metadata,
            args.epochs, args.lr, args.batch_size,
            device, ckpt_dir, plot_dir,
        )
    else:  # head
        run_head_mode(
            rep_dir, train_idx, val_idx, metadata,
            args.epochs, args.lr, args.batch_size,
            device, ckpt_dir, plot_dir,
        )

    print("\nAll done.")
    print(f"Checkpoints → {ckpt_dir}")
    print(f"Plots       → {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train linear probes on extracted LLM representations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--rep_dir", required=True,
        help="Directory from extract_representations.py\n"
             "(contains layer_*.pt / head_*.pt and metadata.json)",
    )
    parser.add_argument(
        "--mode", choices=["layer", "head"], required=True,
        help="layer — one probe per transformer layer\n"
             "head  — one probe per (layer, attention head)",
    )
    parser.add_argument(
        "--output_dir", default="data/probes",
        help="Root directory for checkpoints and plots (default: data/probes)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs per probe (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Mini-batch size during training (default: 64)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="Fraction of questions assigned to validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible train/val split (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training: cuda | cpu (default: auto-detect)",
    )
    args = parser.parse_args()
    main(args)
