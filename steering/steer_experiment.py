"""
Steering experiment: apply a steering vector during inference and measure
how it shifts the model's preference for safe vs. vulnerable code.

Measurement (mirrors head_causal_analysis.py):

    Delta(α) = mean_logP(safe_response | prompt, steered by α)
             − mean_logP(vuln_response | prompt, steered by α)

Higher Delta = model assigns more probability to safe code.
α = 0 is the unsteered baseline.

Steering modes:
  layer  — add α·v to the residual stream output of a transformer layer
  head   — add α·v to the attention head output slice (before o_proj)

Layer / head selection (two options, controlled by whether --layer_idx is given):
  Explicit  — supply --layer_idx (and --head_idx for head mode).
  Auto      — omit --layer_idx; the script reads --probe_results and picks
                · layer mode: the layer with the highest probe accuracy
                · head mode:  top --top_k heads by probe accuracy

Layer index convention: matches the `layer` field in probe accuracy results
  and the numeric suffix in steering vector filenames (e.g. layer_23.pt → 23).

Usage:
  # Layer steering, best layer auto-selected
  python steer_experiment.py \\
      --pairs_file data/contrastive_pairs/llama31-8b_cross.jsonl \\
      --steering_dir data/steering_vectors \\
      --probe_results data/probes/meta-llama_Llama-3_1-8B-Instruct/safe_only/cwe-022/response_mean/layer/plots/layer_accuracy_results.json \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --cwe_id cwe-022 --dataset_type safe_only --token_agg response_mean \\
      --mode layer --method mean_diff \\
      --alphas 0 5 10 20 50 \\
      --output_dir results/steering/cwe-022/layer

  # Head steering, explicit single head
  python steer_experiment.py \\
      --pairs_file data/contrastive_pairs/llama31-8b_cross.jsonl \\
      --steering_dir data/steering_vectors \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --cwe_id cwe-022 --dataset_type safe_only --token_agg response_mean \\
      --mode head --method probe \\
      --layer_idx 23 --head_idx 1 \\
      --alphas 0 5 10 20 50 \\
      --output_dir results/steering/cwe-022/head

  # Head steering, auto top-5 heads
  python steer_experiment.py \\
      --pairs_file data/contrastive_pairs/llama31-8b_cross.jsonl \\
      --steering_dir data/steering_vectors \\
      --probe_results data/probes/meta-llama_Llama-3_1-8B-Instruct/safe_only/cwe-022/response_mean/head/plots/head_accuracy_results.json \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --cwe_id cwe-022 --dataset_type safe_only --token_agg response_mean \\
      --mode head --method probe \\
      --top_k 5 \\
      --alphas 0 5 10 20 50 \\
      --output_dir results/steering/cwe-022/head
"""

from __future__ import annotations

import argparse
import json
import sys
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from prompts import CODE_GENERATION_PROMPT  # single prompt source, rendered at run time


# --------------------------------------------------------------------------- #
# Prompt / data helpers  (shared with head_causal_analysis.py)
# --------------------------------------------------------------------------- #



def build_safe_messages(question: str) -> list[dict]:
    content = CODE_GENERATION_PROMPT.format(question=question)
    return [{"role": "user", "content": content}]


def build_chat_inputs(
    messages: list[dict], response: str, tokenizer
) -> tuple[list[int], int]:
    prefix_ids: list[int] = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    full_ids: list[int] = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False,
    )
    return full_ids, min(len(prefix_ids), len(full_ids))


def prepare_batch(
    pairs: list[dict],
    response_key: str,
    tokenizer,
    device: str,
    model,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    all_ids, n_prefix_list = [], []
    for item in pairs:
        ids, n_prefix = build_chat_inputs(
            item["safe_messages"], item[response_key], tokenizer
        )
        all_ids.append(ids)
        n_prefix_list.append(n_prefix)

    max_len = max(len(ids) for ids in all_ids)
    pad_id  = tokenizer.pad_token_id

    input_ids_padded, attention_masks = [], []
    for ids in all_ids:
        pad_len = max_len - len(ids)
        input_ids_padded.append([pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    input_ids      = torch.tensor(input_ids_padded, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks,  dtype=torch.long)

    if device not in ("auto", "cpu") and not hasattr(model, "hf_device_map"):
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    return {"input_ids": input_ids, "attention_mask": attention_mask}, n_prefix_list


def build_response_mask(
    attention_mask: torch.Tensor,
    n_prefix_list: list[int],
) -> torch.Tensor:
    cpu_mask = attention_mask.cpu()
    batch, seq_len = cpu_mask.shape
    response_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    for i, n_prefix in enumerate(n_prefix_list):
        real_len   = int(cpu_mask[i].sum().item())
        pad_len    = seq_len - real_len
        resp_start = pad_len + n_prefix
        if resp_start < seq_len:
            response_mask[i, resp_start:] = True
    return response_mask


def _cwe_key(cwe_id: str) -> str:
    """Normalize a CWE id for comparison: 'cwe-022', 'CWE-22', '022', '22' -> '22'."""
    return str(cwe_id).lower().removeprefix("cwe-").lstrip("0") or "0"


def load_pairs(pairs_file: Path, cwe_id: str, max_pairs: int | None) -> list[dict]:
    pairs: list[dict] = []
    target_cwe = _cwe_key(cwe_id)
    with open(pairs_file) as f:
        for line in f:
            entry = json.loads(line)
            if _cwe_key(entry.get("cwe_id", "")) != target_cwe:
                continue
            question = entry.get("question", "")
            if not question or not entry.get("safe_code") or not entry.get("vuln_code"):
                continue
            pairs.append({
                "safe_messages": build_safe_messages(question),
                "safe_response": entry["safe_code"],
                "vuln_response": entry["vuln_code"],
                "pair_id":       entry.get("id", f"pair_{len(pairs)}"),
            })
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    return pairs


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_name: str, dtype_str: str, device: str):
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[dtype_str]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": dtype}
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    if device not in ("auto", "cpu") and not hasattr(model, "hf_device_map"):
        model = model.to(device)

    return model, tokenizer


def get_model_head_config(model) -> tuple[int, int]:
    cfg = model.config
    n_heads  = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    return n_heads, head_dim


# --------------------------------------------------------------------------- #
# Log-prob computation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_avg_log_prob_batch(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> list[float]:
    outputs   = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.nn.functional.log_softmax(outputs.logits.float(), dim=-1)

    device        = log_probs.device
    resp_mask_dev = response_mask.to(device)
    results: list[float] = []

    for b in range(input_ids.shape[0]):
        resp_positions = resp_mask_dev[b].nonzero(as_tuple=True)[0]
        valid_positions = resp_positions[resp_positions > 0]
        if len(valid_positions) == 0:
            results.append(float("nan"))
            continue
        token_ids   = input_ids[b, valid_positions]
        gathered_lp = log_probs[b, valid_positions - 1, token_ids]
        results.append(gathered_lp.mean().item())

    return results


# --------------------------------------------------------------------------- #
# Steering hooks
# --------------------------------------------------------------------------- #

class SteerLayer:
    """
    Forward hook on a transformer layer's output that adds α·v to the
    residual stream at response token positions.

    Usage:
        steerer = SteerLayer(model, layer_idx, steering_vec, alpha)
        steerer.response_mask = mask   # (seq,) bool, set before each forward
        model(...)
        steerer.response_mask = None
        steerer.remove()
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        steering_vec: torch.Tensor,  # (hidden_dim,) unit-normed
        alpha: float,
    ):
        self.steering_vec = steering_vec
        self.alpha        = alpha
        self.response_mask: torch.Tensor | None = None

        layer = model.model.layers[layer_idx - 1]  # layer_idx is 1-based
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if self.alpha == 0 or self.response_mask is None:
            return
        hidden = output[0] if isinstance(output, tuple) else output
        mask = self.response_mask.to(hidden.device)
        if not mask.any():
            return
        sv = (self.alpha * self.steering_vec).to(hidden.device)
        hidden = hidden.clone()
        hidden[:, mask, :] += sv
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def remove(self):
        self._handle.remove()


class SteerHead:
    """
    Forward pre-hook on o_proj that adds α·v to one attention head's output
    slice at response token positions.

    Compatible with LLaMA / Mistral / Gemma (any model with an o_proj per layer).

    Usage:
        steerer = SteerHead(model, layer_idx, head_idx, head_dim, steering_vec, alpha)
        steerer.response_mask = mask   # (seq,) bool, set before each forward
        model(...)
        steerer.response_mask = None
        steerer.remove()
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        head_idx: int,
        head_dim: int,
        steering_vec: torch.Tensor,  # (head_dim,) unit-normed
        alpha: float,
    ):
        self.head_idx     = head_idx - 1  # convert to 0-based for slice computation
        self.head_dim     = head_dim
        self.steering_vec = steering_vec
        self.alpha        = alpha
        self.response_mask: torch.Tensor | None = None

        o_proj = self._find_o_proj(model, layer_idx - 1)  # layer_idx is 1-based
        if o_proj is None:
            raise ValueError(f"Cannot find o_proj for layer {layer_idx}.")
        self._handle = o_proj.register_forward_pre_hook(self._hook_fn)

    @staticmethod
    def _find_o_proj(model, layer_idx: int):
        o_proj_modules = [
            m.o_proj for m in model.modules() if hasattr(m, "o_proj")
        ]
        return o_proj_modules[layer_idx] if layer_idx < len(o_proj_modules) else None

    def _hook_fn(self, module, inputs):
        if self.alpha == 0 or self.response_mask is None:
            return
        x = inputs[0]   # (batch, seq, n_heads * head_dim)
        mask = self.response_mask.to(x.device)
        if not mask.any():
            return
        sv    = (self.alpha * self.steering_vec).to(x.device)
        start = self.head_idx * self.head_dim
        end   = start + self.head_dim
        x = x.clone()
        x[:, mask, start:end] += sv
        return (x,)

    def remove(self):
        self._handle.remove()


# --------------------------------------------------------------------------- #
# Steering vector loading
# --------------------------------------------------------------------------- #

def slugify(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def load_layer_sv(
    steering_dir: Path,
    model: str,
    dataset_type: str,
    cwe_id: str,
    token_agg: str,
    method: str,
    layer_idx: int,     # matches probe result `layer` field / filename suffix
) -> torch.Tensor:
    fname = (
        steering_dir / slugify(model) / dataset_type / cwe_id / token_agg
        / "layer" / method / f"steering_vector_layer_{layer_idx:02d}.pt"
    )
    if not fname.exists():
        raise FileNotFoundError(f"Layer steering vector not found: {fname}")
    data = torch.load(fname, map_location="cpu", weights_only=True)
    return data["steering_vector"].float()


def load_head_sv(
    steering_dir: Path,
    model: str,
    dataset_type: str,
    cwe_id: str,
    token_agg: str,
    method: str,
    layer_idx: int,
    head_idx: int,
) -> torch.Tensor:
    fname = (
        steering_dir / slugify(model) / dataset_type / cwe_id / token_agg
        / "head" / method / f"steering_vector_head_{layer_idx:02d}_{head_idx:02d}.pt"
    )
    if not fname.exists():
        raise FileNotFoundError(f"Head steering vector not found: {fname}")
    data = torch.load(fname, map_location="cpu", weights_only=True)
    return data["steering_vector"].float()


# --------------------------------------------------------------------------- #
# Target selection
# --------------------------------------------------------------------------- #

def select_targets(args) -> list[dict]:
    """
    Returns a list of target dicts.  Each has:
      layer_idx      — matches probe result `layer` value / filename suffix
      head_idx       — only present for head mode
      probe_accuracy — float or None
      label          — display string

    Explicit selection (--layer_idx given):
      layer mode → single layer target
      head mode  → single head target (requires --head_idx)

    Auto selection (--layer_idx omitted, requires --probe_results):
      layer mode → best layer by probe accuracy
      head mode  → top --top_k heads by probe accuracy
    """
    if args.layer_idx is not None:
        # --- Explicit ---
        if args.mode == "head":
            if args.head_idx is None:
                raise ValueError("--head_idx is required when --layer_idx is specified in head mode.")
            return [{
                "layer_idx": args.layer_idx,
                "head_idx":  args.head_idx,
                "probe_accuracy": None,
                "label": f"L{args.layer_idx:02d}H{args.head_idx:02d}",
            }]
        else:
            return [{
                "layer_idx": args.layer_idx,
                "probe_accuracy": None,
                "label": f"layer_{args.layer_idx:02d}",
            }]

    # --- Auto ---
    if not args.probe_results:
        raise ValueError(
            "--probe_results is required for auto layer/head selection "
            "(omit --layer_idx to trigger auto mode)."
        )

    with open(args.probe_results) as f:
        results = json.load(f)

    # Auto-detect causal analysis format (dict with "heads" key) vs probe format (flat list)
    is_causal = isinstance(results, dict) and "heads" in results
    if is_causal:
        head_list = results["heads"]  # already sorted by causal_rank ascending
    else:
        head_list = results  # flat list sorted by val_accuracy descending

    if args.mode == "head":
        if is_causal:
            # Select top-k safe-promoting heads (lowest causal_rank = largest safety drop on knockout)
            safe_heads = [h for h in head_list if h.get("delta_vs_baseline", 0) < 0]
            top = safe_heads[:args.top_k]
            return [
                {
                    "layer_idx": r["layer"],
                    "head_idx":  r["head"],
                    "probe_accuracy": abs(r.get("delta_vs_baseline", 0)),
                    "label": f"L{r['layer']:02d}H{r['head']:02d}",
                }
                for r in top
            ]
        else:
            top = head_list[:args.top_k]
            return [
                {
                    "layer_idx": r["layer"],
                    "head_idx":  r["head"],
                    "probe_accuracy": r["val_accuracy"],
                    "label": f"L{r['layer']:02d}H{r['head']:02d}",
                }
                for r in top
            ]
    else:
        if is_causal:
            # Layer mode from causal results: pick layer with biggest total |delta_vs_baseline|
            from collections import defaultdict
            layer_score: defaultdict[int, float] = defaultdict(float)
            for r in head_list:
                layer_score[r["layer"]] += abs(r.get("delta_vs_baseline", 0))
            best_layer = max(layer_score, key=lambda l: layer_score[l])
            best_acc   = layer_score[best_layer]
        elif head_list and "head" in head_list[0]:
            # Head-level probe results — pick layer with highest max head accuracy
            from collections import defaultdict
            layer_max: defaultdict[int, float] = defaultdict(float)
            for r in head_list:
                layer_max[r["layer"]] = max(layer_max[r["layer"]], r["val_accuracy"])
            best_layer = max(layer_max, key=lambda l: layer_max[l])
            best_acc   = layer_max[best_layer]
        else:
            # Layer-level results (list of {layer, val_accuracy})
            best = head_list[0]
            best_layer = best["layer"]
            best_acc   = best["val_accuracy"]

        return [{
            "layer_idx": best_layer,
            "probe_accuracy": best_acc,
            "label": f"layer_{best_layer:02d}",
        }]


# --------------------------------------------------------------------------- #
# Core scoring helper
# --------------------------------------------------------------------------- #

def score_all_pairs(
    model,
    tokenizer,
    pairs: list[dict],
    response_key: str,
    batch_size: int,
    device: str,
    steerer=None,
) -> list[float]:
    """
    Teacher-forced avg log-prob for response_key across all pairs.
    If steerer is not None, its response_mask is set before each batch.
    """
    all_lp: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoding, n_prefix_list = prepare_batch(batch, response_key, tokenizer, device, model)
        resp_mask = build_response_mask(encoding["attention_mask"], n_prefix_list)

        if steerer is not None:
            # Union mask so the hook fires at any response position in the batch;
            # per-sample log-prob still uses the correct per-row resp_mask.
            steerer.response_mask = resp_mask.any(dim=0)  # (seq,)

        lp = compute_avg_log_prob_batch(
            model,
            encoding["input_ids"],
            encoding["attention_mask"],
            resp_mask,
        )
        all_lp.extend(lp)

        if steerer is not None:
            steerer.response_mask = None

    return all_lp


def compute_delta(safe_lp: list[float], vuln_lp: list[float]) -> tuple[float, float]:
    """Mean and std of per-pair (safe_lp − vuln_lp), ignoring NaNs."""
    deltas = [s - v for s, v in zip(safe_lp, vuln_lp)
              if not (math.isnan(s) or math.isnan(v))]
    if not deltas:
        return float("nan"), float("nan")
    return float(np.mean(deltas)), float(np.std(deltas))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_delta_vs_alpha(
    alphas: list[float],
    target_results: list[dict],   # each has 'label', 'alpha_results'[{alpha, mean_delta, std_delta}]
    baseline_delta: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.get_cmap("tab10", max(len(target_results), 1))

    for i, target in enumerate(target_results):
        xs = [r["alpha"] for r in target["alpha_results"]]
        ys = [r["mean_delta"] for r in target["alpha_results"]]
        es = [r["std_delta"]  for r in target["alpha_results"]]
        ax.errorbar(
            xs, ys, yerr=es,
            label=target["label"],
            color=cmap(i),
            linewidth=1.8,
            marker="o",
            markersize=5,
            capsize=3,
        )

    ax.axhline(
        baseline_delta,
        color="black", linewidth=1.2, linestyle="--",
        label=f"baseline Δ = {baseline_delta:+.4f}",
    )
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")

    ax.set_xlabel("Steering coefficient α", fontsize=12)
    ax.set_ylabel("Δ = mean logP(safe) − mean logP(vuln)", fontsize=12)
    ax.set_title(
        "Effect of steering on safe-vs-vulnerable preference\n"
        "(higher Δ = model assigns more probability to safe code)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #

def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Select targets ----
    targets = select_targets(args)
    print(f"Targets ({len(targets)}):")
    for t in targets:
        acc_str = f"  probe_acc={t['probe_accuracy']:.4f}" if t["probe_accuracy"] else ""
        print(f"  {t['label']}{acc_str}")

    # ---- Load data ----
    print(f"\nLoading pairs for {args.cwe_id} from {args.pairs_file} ...")
    pairs = load_pairs(Path(args.pairs_file), args.cwe_id, args.max_pairs)
    if not pairs:
        raise ValueError(f"No pairs found for {args.cwe_id}.")
    print(f"  {len(pairs)} pairs loaded.")

    # ---- Load model ----
    print(f"\nLoading model {args.model} ...")
    model, tokenizer = load_model(args.model, args.dtype, args.device)
    n_heads, head_dim = get_model_head_config(model)
    print(f"  n_heads={n_heads}  head_dim={head_dim}")

    steering_dir = Path(args.steering_dir)

    # ---- Baseline (α = 0, no steering) ----
    print("\nComputing baseline (α = 0) ...")
    base_safe_lp = score_all_pairs(model, tokenizer, pairs, "safe_response",
                                   args.batch_size, args.device)
    base_vuln_lp = score_all_pairs(model, tokenizer, pairs, "vuln_response",
                                   args.batch_size, args.device)
    baseline_delta, baseline_std = compute_delta(base_safe_lp, base_vuln_lp)
    print(f"  Baseline Δ = {baseline_delta:+.6f}  std = {baseline_std:.6f}")
    print(f"  (positive → model already prefers safe code under benign prompt)")

    # ---- Alpha sweep per target ----
    # Filter out α=0; it is already computed as the baseline.
    alphas_nonzero = [a for a in args.alphas if a != 0.0]

    all_target_results: list[dict] = []

    for t_idx, target in enumerate(targets):
        layer_idx = target["layer_idx"]
        head_idx  = target.get("head_idx")
        label     = target["label"]
        print(f"\n[{t_idx+1}/{len(targets)}] {label} ...")

        # Load steering vector once per target
        if args.mode == "head":
            sv = load_head_sv(
                steering_dir, args.model, args.dataset_type, args.cwe_id,
                args.token_agg, args.method, layer_idx, head_idx,
            )
        else:
            sv = load_layer_sv(
                steering_dir, args.model, args.dataset_type, args.cwe_id,
                args.token_agg, args.method, layer_idx,
            )
        print(f"  Steering vector loaded  shape={tuple(sv.shape)}  "
              f"norm={sv.norm().item():.4f}")

        alpha_results: list[dict] = [{
            "alpha":       0.0,
            "mean_delta":  baseline_delta,
            "std_delta":   baseline_std,
            "delta_vs_baseline": 0.0,
        }]

        for alpha in alphas_nonzero:
            # Create hook
            if args.mode == "head":
                steerer = SteerHead(
                    model, layer_idx, head_idx, head_dim, sv, alpha
                )
            else:
                steerer = SteerLayer(model, layer_idx, sv, alpha)

            safe_lp = score_all_pairs(model, tokenizer, pairs, "safe_response",
                                      args.batch_size, args.device, steerer)
            vuln_lp = score_all_pairs(model, tokenizer, pairs, "vuln_response",
                                      args.batch_size, args.device, steerer)
            steerer.remove()

            mean_d, std_d = compute_delta(safe_lp, vuln_lp)
            print(f"  α={alpha:+6.1f}  Δ={mean_d:+.6f}  std={std_d:.6f}  "
                  f"vs_baseline={mean_d - baseline_delta:+.6f}")

            alpha_results.append({
                "alpha":             alpha,
                "mean_delta":        mean_d,
                "std_delta":         std_d,
                "delta_vs_baseline": mean_d - baseline_delta,
            })

        # Sort by alpha for clean output
        alpha_results.sort(key=lambda x: x["alpha"])

        all_target_results.append({
            "label":         label,
            "layer_idx":     layer_idx,
            "head_idx":      head_idx,
            "probe_accuracy": target["probe_accuracy"],
            "alpha_results": alpha_results,
        })

    # ---- Save JSON ----
    results_path = output_dir / "steering_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model":          args.model,
            "cwe_id":         args.cwe_id,
            "dataset_type":   args.dataset_type,
            "token_agg":      args.token_agg,
            "mode":           args.mode,
            "method":         args.method,
            "n_pairs":        len(pairs),
            "baseline_delta": baseline_delta,
            "baseline_std":   baseline_std,
            "alphas":         sorted(set([0.0] + args.alphas)),
            "targets":        all_target_results,
        }, f, indent=2)
    print(f"\nSaved results → {results_path}")

    # ---- Plot ----
    plot_path = output_dir / "delta_vs_alpha.png"
    plot_delta_vs_alpha(args.alphas, all_target_results, baseline_delta, plot_path)

    # ---- Summary ----
    print("\n--- Summary ---")
    print(f"{'Target':<14}  {'α':>6}  {'Δ':>10}  {'vs_base':>10}")
    print(f"  (baseline)       {0:>6.1f}  {baseline_delta:>+10.6f}  {'—':>10}")
    for t in all_target_results:
        for r in t["alpha_results"]:
            if r["alpha"] == 0.0:
                continue
            print(f"  {t['label']:<12}  {r['alpha']:>6.1f}  "
                  f"{r['mean_delta']:>+10.6f}  {r['delta_vs_baseline']:>+10.6f}")

    print("\nDone.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steering vector experiment: measure effect on safe-vs-vuln preference.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Data
    parser.add_argument("--pairs_file", required=True,
        help="Contrastive pairs JSONL (cross_group recommended)")
    parser.add_argument("--cwe_id", required=True,
        help="CWE to filter pairs on (e.g. cwe-022)")
    # Model
    parser.add_argument("--model", required=True,
        help="HuggingFace model name")
    parser.add_argument("--dtype", default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)")
    parser.add_argument("--device", default="cuda",
        help="Device: cuda / cpu / auto (default: cuda)")
    # Steering vectors
    parser.add_argument("--steering_dir", default="data/steering_vectors",
        help="Root steering vectors directory (default: data/steering_vectors)")
    parser.add_argument("--dataset_type", default="safe_only",
        help="Dataset type matching steering vectors (default: safe_only)")
    parser.add_argument("--token_agg", default="response_mean",
        choices=["response_last", "response_mean"],
        help="Token aggregation matching steering vectors (default: response_mean)")
    parser.add_argument("--method", default="mean_diff",
        choices=["mean_diff", "probe"],
        help="Steering vector method to use (default: mean_diff)")
    # Mode
    parser.add_argument("--mode", default="layer",
        choices=["layer", "head"],
        help="Steer at layer residual stream or head output (default: layer)")
    # Explicit target selection
    parser.add_argument("--layer_idx", type=int, default=None,
        help="Layer index (matches probe result `layer` field / filename suffix).\n"
             "If omitted, auto-select via --probe_results.")
    parser.add_argument("--head_idx", type=int, default=None,
        help="Head index (1-based). Required when --layer_idx is set in head mode.")
    # Auto target selection
    parser.add_argument("--probe_results", default=None,
        help="Path to probe accuracy JSON:\n"
             "  head mode  → head_accuracy_results.json\n"
             "  layer mode → layer_accuracy_results.json (or head results, best layer derived)\n"
             "Required when --layer_idx is omitted.")
    parser.add_argument("--top_k", type=int, default=5,
        help="Number of top-probe heads to steer (head mode auto only, default: 5)")
    # Experiment
    parser.add_argument("--alphas", type=float, nargs="+", default=[1, 2, 3, 5, 10],
        help="Steering coefficient(s) α to sweep (default: 0 5 10 20 50)")
    parser.add_argument("--max_pairs", type=int, default=None,
        help="Cap on number of pairs to process (default: all)")
    parser.add_argument("--batch_size", type=int, default=4,
        help="Batch size for model forward passes (default: 4)")
    # Output
    parser.add_argument("--output_dir", required=True,
        help="Directory to save results JSON and plot")

    run_experiment(parser.parse_args())
