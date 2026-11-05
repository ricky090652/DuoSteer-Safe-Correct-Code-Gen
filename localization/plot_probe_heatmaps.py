"""
Redraw linear-probe heatmaps in the style of the paper's Figure 1.

Reads only the JSON files written by train_probe.py (no retraining):
  {probe_dir}/cwe-XXX/{token_agg}/plots/head_accuracy_results.json
  {probe_dir}/cwe-XXX/{token_agg}/plots/layer_accuracy_results.json   (optional)

Each panel shows per-head accuracy at its real (layer, head) position — heads
are NOT sorted — with layer 1 at the bottom, a layer-level strip ("L") on the
right, the best head and best layer boxed, and a red→yellow→green colormap on a
shared fixed scale (paper: 55–85%).

Outputs:
  {probe_dir}/cwe-XXX/{token_agg}/plots/head_heatmap.png   (one per CWE)
  {probe_dir}/probe_heatmap_all_cwes.png                   (all CWEs side by side)

Usage:
  python localization/plot_probe_heatmaps.py \\
      --probe_dir data/probes/Qwen_Qwen2_5-Coder-7B-Instruct/qwen25-coder-7b_intra
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_head_matrix(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (n_layers, n_heads) accuracy matrix (row 0 = layer 1) and best (layer, head), 1-based."""
    rows = json.loads(path.read_text())
    n_layers = max(r["layer"] for r in rows)
    n_heads = max(r["head"] for r in rows)
    mat = np.full((n_layers, n_heads), np.nan)
    for r in rows:
        mat[r["layer"] - 1, r["head"] - 1] = r["val_accuracy"]
    best = max(rows, key=lambda r: r["val_accuracy"])
    return mat, (best["layer"], best["head"])


def load_layer_vector(path: Path, n_layers: int) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Return per-layer accuracy (index 0 = layer 1) and the best layer, or (None, None)."""
    if not path.exists():
        return None, None
    rows = json.loads(path.read_text())
    # Keep only transformer layers 1..n_layers (drops an embedding layer 0 if present).
    rows = [r for r in rows if 1 <= r["layer"] <= n_layers]
    vec = np.full(n_layers, np.nan)
    for r in rows:
        vec[r["layer"] - 1] = r["val_accuracy"]
    best = max(rows, key=lambda r: r["val_accuracy"])["layer"] if rows else None
    return vec, best


def axis_ticks(n: int) -> list[int]:
    """1, 8, 16, ... , n  (matches the paper's tick style)."""
    ticks = [1] + list(range(8, n, 8))
    if ticks[-1] != n:
        ticks.append(n)
    return ticks


def draw_panel(
    ax,
    head_mat: np.ndarray,
    best_head: tuple[int, int],
    layer_vec: Optional[np.ndarray],
    best_layer: Optional[int],
    norm: Normalize,
    cmap,
    title: str = "",
    show_ylabel: bool = True,
):
    """Draw one CWE panel (percent scale). Returns the head image for the colorbar."""
    n_layers, n_heads = head_mat.shape
    extent = (0.5, n_heads + 0.5, 0.5, n_layers + 0.5)
    im = ax.imshow(head_mat * 100, cmap=cmap, norm=norm, origin="lower",
                   extent=extent, aspect="auto", interpolation="nearest")
    l, h = best_head
    ax.add_patch(Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False,
                           edgecolor="black", linewidth=1.3))
    ax.set_xticks(axis_ticks(n_heads))
    ax.set_yticks(axis_ticks(n_layers))
    ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("Head", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Layer", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)

    if layer_vec is not None:
        strip = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.06)
        strip.imshow((layer_vec * 100)[:, None], cmap=cmap, norm=norm, origin="lower",
                     extent=(0.5, 1.5, 0.5, n_layers + 0.5), aspect="auto",
                     interpolation="nearest")
        if best_layer is not None:
            strip.add_patch(Rectangle((0.5, best_layer - 0.5), 1, 1, fill=False,
                                      edgecolor="black", linewidth=1.3))
        strip.set_yticks([])
        strip.set_xticks([1])
        strip.set_xticklabels(["L"])
        strip.tick_params(labelsize=7, length=0)
    return im


def plot_single(head_mat, best_head, layer_vec, best_layer, norm, cmap, title, out_path: Path):
    n_layers, n_heads = head_mat.shape
    fig, ax = plt.subplots(figsize=(max(4.5, n_heads * 0.16), max(4.0, n_layers * 0.15)))
    im = draw_panel(ax, head_mat, best_head, layer_vec, best_layer, norm, cmap, title)
    cbar = fig.colorbar(im, ax=fig.axes, fraction=0.04, pad=0.03)
    cbar.set_label("Probe accuracy (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main(args):
    probe_dir = Path(args.probe_dir)
    cmap = plt.get_cmap(args.cmap)
    norm = Normalize(vmin=args.vmin, vmax=args.vmax)

    panels = []
    for cwe in args.cwes:
        plot_dir = probe_dir / f"cwe-{cwe}" / args.token_agg / "plots"
        head_json = plot_dir / "head_accuracy_results.json"
        if not head_json.exists():
            print(f"[skip] CWE-{cwe}: {head_json} not found")
            continue
        head_mat, best_head = load_head_matrix(head_json)
        layer_vec, best_layer = load_layer_vector(
            plot_dir / "layer_accuracy_results.json", head_mat.shape[0])
        n_hi = int((head_mat >= args.threshold).sum())
        print(f"CWE-{cwe}: best head L{best_head[0]}H{best_head[1]} "
              f"({np.nanmax(head_mat):.3f}), best layer {best_layer}, "
              f"heads ≥ {args.threshold}: {n_hi}")
        plot_single(head_mat, best_head, layer_vec, best_layer, norm, cmap,
                    f"CWE-{cwe}", plot_dir / args.single_name)
        panels.append((cwe, head_mat, best_head, layer_vec, best_layer))

    if not panels:
        return

    n_layers, n_heads = panels[0][1].shape
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.0 * len(panels) + 0.8, 3.0 * n_layers / n_heads + 0.4))
    axes = np.atleast_1d(axes)
    for i, (ax, (cwe, hm, bh, lv, bl)) in enumerate(zip(axes, panels)):
        im = draw_panel(ax, hm, bh, lv, bl, norm, cmap, f"CWE-{cwe}", show_ylabel=(i == 0))
    cbar = fig.colorbar(im, ax=fig.axes, fraction=0.012, pad=0.015)
    cbar.set_label("Probe accuracy (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    out = Path(args.out) if args.out else probe_dir / "probe_heatmap_all_cwes.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Redraw probe heatmaps (paper Figure 1 style).",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--probe_dir", required=True,
                   help="Directory holding cwe-XXX/ subfolders (train_probe.py --output_dir parent)")
    p.add_argument("--cwes", nargs="+", default=["022", "079", "094", "295", "502"])
    p.add_argument("--token_agg", default="response_mean")
    p.add_argument("--vmin", type=float, default=55.0, help="Colormap min in %% (paper: 55)")
    p.add_argument("--vmax", type=float, default=85.0, help="Colormap max in %% (paper: 85)")
    p.add_argument("--cmap", default="RdYlGn", help="Low = red, high = green (paper style)")
    p.add_argument("--threshold", type=float, default=0.65,
                   help="Report the number of heads at or above this accuracy (paper §5 uses 0.65)")
    p.add_argument("--single_name", default="head_heatmap.png",
                   help="Per-CWE output filename inside each plots/ dir")
    p.add_argument("--out", default=None, help="Combined figure path")
    main(p.parse_args())
