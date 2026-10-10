"""
Train and evaluate the CWE router on prompt representations.

Input: the output of extract_prompt_representations.py
  {rep_dir}/train.pt, train.jsonl, eval.pt, eval.jsonl

Classifier: multinomial logistic regression (nn.Linear(d, 5) + cross-entropy)
with an L2 penalty on W, fitted full-batch with L-BFGS on standardized
features. Classes are re-weighted to balance the loss by default.

Model selection (train set only):
  for every layer and every L2 strength, k-fold cross-validation with
  question-level folds stratified by CWE; pick the (layer, L2) with the best
  mean CV balanced accuracy. The selected config is refitted on the whole
  train set and scored once on the eval set (SecCodePLT).

Also reported, for analysis only (never used for selection):
  per-layer eval accuracy (each layer at its best CV L2) and a keyword
  baseline built from generic CWE vocabulary.

Output  {out_dir}/
  results.json             selection grid, chosen config, eval metrics, keyword baseline
  eval_predictions.jsonl   per-task prediction and class probabilities
  router.pt                {W, b, mean, std, layer (1-based), classes, l2}
  layer_curve.png          per-layer CV / eval balanced accuracy

Usage:
  python localization/train_router.py --model Qwen/Qwen2.5-Coder-7B-Instruct
  # Qwen-labelled questions only
  python localization/train_router.py --model Qwen/Qwen2.5-Coder-7B-Instruct \\
      --train_origins qwen25-coder-7b
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from common.utils import read_jsonl, write_jsonl

CWES = ["022", "079", "094", "295", "502"]

# Generic CWE vocabulary; ties go to the earlier (more specific) class.
KEYWORDS = {
    "502": r"deserializ|serializ|pickle|unpickl|yaml|marshal|jsonpickle|shelve",
    "295": r"\bssl\b|\btls\b|certificate|https|sftp|smtp|\bftps?\b|paramiko|\bldaps?\b|\bssh\b",
    "094": r"\beval|\bexec|expression|snippet|arithmetic|calculat|python code|script",
    "079": r"html|web ?page|render|browser|markup|xss|<\w+>",
    "022": r"\bpaths?\b|file ?path|director|folder|filename",
}


def slugify(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_split(rep_dir: Path, name: str):
    reps = torch.load(rep_dir / f"{name}.pt")["reps"]
    items = read_jsonl(rep_dir / f"{name}.jsonl")
    assert reps.shape[0] == len(items), f"{name}: rep/item count mismatch"
    y = torch.tensor([CWES.index(it["cwe_id"]) for it in items])
    return reps, items, y


def make_folds(items: list[dict], y: torch.Tensor, k: int, seed: int) -> list[torch.Tensor]:
    """Question-level folds (by src_id), stratified by CWE. Returns per-row fold ids."""
    group_label, group_rows = {}, defaultdict(list)
    for i, it in enumerate(items):
        group_rows[it["src_id"]].append(i)
        group_label.setdefault(it["src_id"], int(y[i]))
    rng = random.Random(seed)
    fold = torch.empty(len(items), dtype=torch.long)
    for c in range(len(CWES)):
        groups = sorted(g for g, lab in group_label.items() if lab == c)
        rng.shuffle(groups)
        for j, g in enumerate(groups):
            fold[group_rows[g]] = j % k
    return fold


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #

def standardize(train_x: torch.Tensor, *others: torch.Tensor):
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-6)
    return (mean, std), [(x - mean) / std for x in (train_x, *others)]


def class_weights(y: torch.Tensor, balanced: bool) -> torch.Tensor:
    counts = torch.bincount(y, minlength=len(CWES)).float()
    if not balanced:
        return torch.ones(len(CWES))
    return counts.sum() / (len(CWES) * counts.clamp_min(1))


def fit(x: torch.Tensor, y: torch.Tensor, l2: float, weight: torch.Tensor,
        max_iter: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-batch L-BFGS for weighted softmax regression with L2 on W."""
    W = torch.zeros(x.shape[1], len(CWES), device=x.device, requires_grad=True)
    b = torch.zeros(len(CWES), device=x.device, requires_grad=True)
    weight = weight.to(x.device)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ W + b, y, weight=weight) + 0.5 * l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def predict_proba(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x @ W + b, dim=1)


def metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """y_pred may contain -1 (abstain), which counts as wrong."""
    cm = torch.zeros(len(CWES), len(CWES) + 1, dtype=torch.long)   # last column = none
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p if p >= 0 else len(CWES)] += 1
    recall = cm[:, :len(CWES)].diag().float() / cm.sum(1).clamp_min(1)
    return {
        "accuracy": (y_true == y_pred).float().mean().item(),
        "balanced_accuracy": recall.mean().item(),
        "recall": {c: round(r, 4) for c, r in zip(CWES, recall.tolist())},
        "confusion": {"rows_true": CWES, "cols_pred": CWES + ["none"],
                      "matrix": cm.tolist()},
    }


def keyword_predict(question: str) -> int:
    text = question.lower()
    scores = [len(re.findall(KEYWORDS[c], text)) for c in KEYWORDS]
    if max(scores) == 0:
        return -1
    return CWES.index(list(KEYWORDS)[scores.index(max(scores))])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    slug = slugify(args.model)
    rep_dir = Path(args.rep_dir) if args.rep_dir else \
        BASE / "data" / "representations" / slug / "router_prompts"
    tag = "all" if not args.train_origins else "+".join(args.train_origins)
    out_dir = Path(args.out_dir) if args.out_dir else \
        BASE / "data" / "probes" / slug / "router" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    tr_reps, tr_items, tr_y = load_split(rep_dir, "train")
    ev_reps, ev_items, ev_y = load_split(rep_dir, "eval")
    if args.train_origins:
        keep = [i for i, it in enumerate(tr_items) if it["label_origin"] in args.train_origins]
        tr_reps, tr_y = tr_reps[keep], tr_y[keep]
        tr_items = [tr_items[i] for i in keep]
    n_layers = tr_reps.shape[1]
    layers = args.layers or list(range(1, n_layers + 1))
    print(f"train {len(tr_items)}  {torch.bincount(tr_y, minlength=5).tolist()}  "
          f"eval {len(ev_items)}  {torch.bincount(ev_y, minlength=5).tolist()}  "
          f"layers {layers[0]}..{layers[-1]}  -> {out_dir}")

    fold = make_folds(tr_items, tr_y, args.n_folds, args.seed)
    tr_y_d, ev_y_d = tr_y.to(device), ev_y.to(device)

    grid = []                       # one row per (layer, l2)
    layer_curve = []                # per layer: best CV l2, refit, eval score
    for layer in layers:
        x_all = tr_reps[:, layer - 1].float().to(device)
        best = None
        for l2 in args.l2_grid:
            bal, acc = [], []
            for f in range(args.n_folds):
                tr_m, va_m = (fold != f).to(device), (fold == f).to(device)
                _, (xt, xv) = standardize(x_all[tr_m], x_all[va_m])
                W, b = fit(xt, tr_y_d[tr_m], l2,
                           class_weights(tr_y[tr_m.cpu()], args.balanced), args.max_iter)
                m = metrics(tr_y[va_m.cpu()], predict_proba(xv, W, b).argmax(1).cpu())
                bal.append(m["balanced_accuracy"])
                acc.append(m["accuracy"])
            row = {"layer": layer, "l2": l2,
                   "cv_balanced_accuracy": sum(bal) / len(bal),
                   "cv_accuracy": sum(acc) / len(acc)}
            grid.append(row)
            if best is None or (row["cv_balanced_accuracy"], row["cv_accuracy"]) > \
                    (best["cv_balanced_accuracy"], best["cv_accuracy"]):
                best = row

        # Refit this layer at its best L2 on all train data (analysis curve).
        x_ev = ev_reps[:, layer - 1].float().to(device)
        _, (xt, xe) = standardize(x_all, x_ev)
        W, b = fit(xt, tr_y_d, best["l2"], class_weights(tr_y, args.balanced), args.max_iter)
        m = metrics(ev_y, predict_proba(xe, W, b).argmax(1).cpu())
        layer_curve.append({**best, "eval_accuracy": m["accuracy"],
                            "eval_balanced_accuracy": m["balanced_accuracy"]})
        print(f"  L{layer:02d}  best l2={best['l2']:<7g} "
              f"cv bal={best['cv_balanced_accuracy']:.3f} acc={best['cv_accuracy']:.3f} | "
              f"eval bal={m['balanced_accuracy']:.3f} acc={m['accuracy']:.3f}")

    # ---- Selected config: chosen on CV only, scored once on eval ----
    chosen = max(grid, key=lambda r: (r["cv_balanced_accuracy"], r["cv_accuracy"]))
    layer, l2 = chosen["layer"], chosen["l2"]
    x_tr = tr_reps[:, layer - 1].float().to(device)
    x_ev = ev_reps[:, layer - 1].float().to(device)
    (mean, std), (xt, xe) = standardize(x_tr, x_ev)
    W, b = fit(xt, tr_y_d, l2, class_weights(tr_y, args.balanced), args.max_iter)
    proba = predict_proba(xe, W, b).cpu()
    pred = proba.argmax(1)
    ev_metrics = metrics(ev_y, pred)
    train_metrics = metrics(tr_y, predict_proba(xt, W, b).argmax(1).cpu())

    kw_ev_pred = [keyword_predict(it["question"]) for it in ev_items]
    kw_eval = metrics(ev_y, torch.tensor(kw_ev_pred))
    kw_train = metrics(tr_y, torch.tensor([keyword_predict(it["question"]) for it in tr_items]))

    print(f"\nselected: layer {layer}, l2 {l2:g}  "
          f"(cv bal={chosen['cv_balanced_accuracy']:.3f})")
    print(f"eval   router  acc={ev_metrics['accuracy']:.3f}  "
          f"bal={ev_metrics['balanced_accuracy']:.3f}  recall={ev_metrics['recall']}")
    print(f"eval   keyword acc={kw_eval['accuracy']:.3f}  "
          f"bal={kw_eval['balanced_accuracy']:.3f}  recall={kw_eval['recall']}")
    print("eval confusion (rows=true, cols=pred " + " ".join(CWES) + " none):")
    for c, row in zip(CWES, ev_metrics["confusion"]["matrix"]):
        print(f"  {c}  " + " ".join(f"{v:4d}" for v in row))

    torch.save({"W": W.cpu(), "b": b.cpu(), "mean": mean.cpu(), "std": std.cpu(),
                "layer": layer, "l2": l2, "classes": CWES,
                "model": args.model, "train_origins": args.train_origins or "all"},
               out_dir / "router.pt")
    write_jsonl([{"id": it.get("src_id"), "cwe_id": it["cwe_id"],
                  "pred_cwe": CWES[int(p)], "correct": it["cwe_id"] == CWES[int(p)],
                  "proba": {c: round(v, 4) for c, v in zip(CWES, pr.tolist())},
                  "keyword_pred": CWES[k] if k >= 0 else "none"}
                 for it, p, pr, k in zip(ev_items, pred, proba, kw_ev_pred)],
                out_dir / "eval_predictions.jsonl")
    results = {
        "model": args.model, "rep_dir": str(rep_dir), "train_origins": args.train_origins or "all",
        "n_train": len(tr_items), "n_eval": len(ev_items),
        "train_counts": dict(zip(CWES, torch.bincount(tr_y, minlength=5).tolist())),
        "config": {"n_folds": args.n_folds, "seed": args.seed, "l2_grid": args.l2_grid,
                   "balanced": args.balanced, "max_iter": args.max_iter},
        "selected": chosen,
        "eval": ev_metrics, "train_fit": train_metrics,
        "keyword_baseline": {"eval": kw_eval, "train": kw_train},
        "layer_curve": layer_curve, "grid": grid,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["layer"] for r in layer_curve]
        plt.figure(figsize=(7, 3.5))
        plt.plot(xs, [r["cv_balanced_accuracy"] for r in layer_curve], "o-", label="CV (train)")
        plt.plot(xs, [r["eval_balanced_accuracy"] for r in layer_curve], "s--",
                 label="SecCodePLT (analysis only)")
        plt.axhline(kw_eval["balanced_accuracy"], color="gray", ls=":", label="keyword (eval)")
        plt.axvline(layer, color="red", alpha=0.3)
        plt.xlabel("layer"); plt.ylabel("balanced accuracy"); plt.ylim(0, 1.02)
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "layer_curve.png", dpi=150)
    except ImportError:
        pass
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                   help="Only used to locate rep_dir / out_dir.")
    p.add_argument("--rep_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--train_origins", nargs="+", default=None,
                   help="Keep train questions whose label_origin is in this list (default: all).")
    p.add_argument("--layers", type=int, nargs="+", default=None, help="1-based; default all.")
    p.add_argument("--l2_grid", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--no_balanced", dest="balanced", action="store_false",
                   help="Disable class re-weighting of the loss.")
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
