"""
Causal head intervention analysis.

For each contrastive pair, knockout one attention head at a time and measure:

    Delta = (1/N_safe) * sum_n logP(r_safe_n | I_ori, head knocked out)
          - (1/N_vuln) * sum_n logP(r_vuln_n | I_ori, head knocked out)

where:
  I_ori   = benign prompt reconstructed from the pair's question (no CWE mention)
  r_safe  = safe_code (code generated under the safe prompt)
  r_vuln  = vuln_code (code generated under the vuln prompt)
  N_*     = number of response tokens (length normalization)

Both responses are scored under the SAME prompt I_ori; the head is zeroed out
at every response token position during scoring.

Interpretation:
  Delta (baseline, no intervention) > 0  →  model prefers safe over vuln code
  After knockout: smaller Delta (ideally negative)  →  head was crucial for
  maintaining that preference  →  more causally important for secure generation

Prompt construction:
  I_ori is rebuilt from scratch using CODE_GENERATION_PROMPT (benign) with the
  pair's question.  Stored messages are NOT used.

Usage:
  python head_causal_analysis.py \\
      --pairs_file  data/contrastive_pairs/llama31-8b_cross.jsonl \\
      --head_results path/to/head_accuracy_results.json \\
      --model       meta-llama/Meta-Llama-3.1-8B-Instruct \\
      --cwe_id      cwe-089 \\
      --output_dir  results/causal_analysis/cwe-089 \\
      --top_k       16 \\
      --max_pairs   200

  # Quick test with a small model
  python head_causal_analysis.py \\
      --pairs_file  data/contrastive_pairs/llama31-8b_cross.jsonl \\
      --head_results path/to/head_accuracy_results.json \\
      --model       huggyllama/llama-7b \\
      --cwe_id      cwe-089 \\
      --output_dir  results/causal_analysis/test \\
      --top_k       4 \\
      --max_pairs   20 \\
      --device      cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import math
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
# CWE database  (kept for --cwe_db arg / future use)
# --------------------------------------------------------------------------- #

def load_cwe_db(cwe_json_path: Path) -> dict:
    with open(cwe_json_path) as f:
        return json.load(f)


def _cwe_numeric_key(cwe_id: str) -> str:
    s = cwe_id.lower().removeprefix("cwe-").lstrip("0")
    return s or "0"


def build_safe_messages(question: str) -> list[dict]:
    """Construct the benign (I_ori) user message for a question."""
    content = CODE_GENERATION_PROMPT.format(question=question)
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# Chat formatting
# --------------------------------------------------------------------------- #

def build_chat_inputs(
    messages: list[dict],
    response: str,
    tokenizer,
) -> tuple[list[int], int]:
    """
    Tokenize [messages + assistant response] as a full chat sequence.

    Returns:
        full_ids  — token ids for the complete sequence
        n_prefix  — number of prefix tokens before the response begins
    """
    prefix_ids: list[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_messages = messages + [{"role": "assistant", "content": response}]
    full_ids: list[int] = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    n_prefix = min(len(prefix_ids), len(full_ids))
    return full_ids, n_prefix


def prepare_batch(
    pairs: list[dict],
    response_key: str,       # "safe_response" or "vuln_response"
    tokenizer,
    device: str,
    model,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """
    Left-pad a batch of [I_ori + response] sequences.
    Always uses safe_messages (I_ori) as the prompt.
    response_key selects which response to score.
    """
    all_ids: list[list[int]] = []
    n_prefix_list: list[int] = []

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
        attention_masks.append([0]      * pad_len + [1] * len(ids))

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
    """
    True for positions belonging to the response tokens.
    With left-padding: response starts at (pad_len + n_prefix).
    """
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


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_name: str, dtype_str: str, device: str):
    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
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


# --------------------------------------------------------------------------- #
# Head knockout hook
# --------------------------------------------------------------------------- #

class HeadKnockout:
    """
    Registers a forward pre-hook on the o_proj of a specific transformer layer
    that zeros out one attention head's output slice at all response token positions.

    Compatible with LLaMA / Mistral / Gemma (any model with an 'o_proj').
    Set `response_mask` (shape: batch, seq) before each forward pass; reset to None after.
    Call `.remove()` to clean up the hook.
    """

    def __init__(self, model, layer_idx: int, head_idx: int, n_heads: int, head_dim: int):
        self.head_idx      = head_idx - 1  # convert to 0-based for slice computation
        self.n_heads       = n_heads
        self.head_dim      = head_dim
        self.response_mask: torch.Tensor | None = None  # (batch, seq) bool

        o_proj = self._find_o_proj(model, layer_idx - 1)  # layer_idx is 1-based
        if o_proj is None:
            raise ValueError(
                f"Could not find o_proj for layer {layer_idx}. "
                "Model architecture may not be supported."
            )
        self._hook_handle = o_proj.register_forward_pre_hook(self._hook_fn)

    @staticmethod
    def _find_o_proj(model, layer_idx: int):
        o_proj_modules = []
        for module in model.modules():
            if hasattr(module, "o_proj"):
                o_proj_modules.append(module.o_proj)
        if layer_idx < len(o_proj_modules):
            return o_proj_modules[layer_idx]
        return None

    def _hook_fn(self, module, inputs):
        """
        inputs[0]: (batch, seq, n_heads * head_dim)
        Zero out the slice for head_idx at each sample's own response positions.
        mask is (batch, seq) so each row is applied only to its own sequence.
        """
        x = inputs[0]
        if self.response_mask is None:
            return

        mask = self.response_mask.to(x.device)  # (batch, seq)
        if not mask.any():
            return

        start = self.head_idx * self.head_dim
        end   = start + self.head_dim

        x = x.clone()
        # mask unsqueezed to (batch, seq, 1) broadcasts over head_dim
        x[:, :, start:end] = x[:, :, start:end].masked_fill(mask.unsqueeze(-1), 0.0)

        return (x,)

    def remove(self):
        self._hook_handle.remove()


# --------------------------------------------------------------------------- #
# Log-prob computation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_avg_log_prob_batch(
    model,
    input_ids: torch.Tensor,       # (batch, seq)
    attention_mask: torch.Tensor,  # (batch, seq)
    response_mask: torch.Tensor,   # (batch, seq) bool, CPU
) -> list[float]:
    """
    Teacher-forced avg log-prob of response tokens for each item in the batch.
    For response token at position p, uses logit at p-1.
    """
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
# Data loading
# --------------------------------------------------------------------------- #

def load_pairs(
    pairs_file: Path,
    cwe_id: str,
    max_pairs: int | None,
) -> list[dict]:
    """
    Load cross-group contrastive pairs for a specific CWE.

    Reconstructs I_ori (benign prompt) from the pair's question.
    Loads both safe_code and vuln_code from the flat pair records.

    When max_pairs is set, CWE-specific sources (codelmsec, cyberseceval-instruct,
    securityeval) are selected first; general sources (emergent-misalignment) fill
    the remainder. This ensures the causal signal comes primarily from questions
    explicitly designed around the target CWE.

    Returns list of dicts with keys:
      safe_messages   — reconstructed benign prompt (I_ori)
      safe_response   — r_safe: code from safe_code
      vuln_response   — r_vuln: code from vuln_code
      pair_id
    """
    # Sources whose questions are explicitly designed for a specific CWE.
    CWE_SPECIFIC_SOURCES = {"codelmsec", "cyberseceval-instruct", "securityeval"}

    specific: list[dict] = []
    general:  list[dict] = []
    target_cwe = _cwe_numeric_key(cwe_id)

    with open(pairs_file) as f:
        for line in f:
            entry = json.loads(line)
            if _cwe_numeric_key(entry.get("cwe_id", "")) != target_cwe:
                continue

            question = entry.get("question", "")
            if not question or not entry.get("safe_code") or not entry.get("vuln_code"):
                continue

            record = {
                "safe_messages": build_safe_messages(question),
                "safe_response": entry["safe_code"],
                "vuln_response": entry["vuln_code"],
                "pair_id":       entry.get("id", ""),
            }
            src = entry.get("source", "")
            if src in CWE_SPECIFIC_SOURCES:
                specific.append(record)
            else:
                general.append(record)

    # CWE-specific questions first, general fill remainder up to max_pairs.
    pairs = specific + general
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    return pairs


def load_top_heads(head_results_path: Path, top_k: int) -> list[dict]:
    with open(head_results_path) as f:
        results = json.load(f)
    return results[:top_k]


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _plot_heatmap(
    head_results: list[dict],
    metric_key: str,
    colorbar_label: str,
    title: str,
    out_path: Path,
    top_n_circled: int = 64,
    vcenter: float = 0.0,
) -> None:
    """
    Generic heatmap helper over (layer × head-index) for any scalar metric.
    Top-N heads by ascending metric value are circled in white.
    vcenter sets the neutral midpoint of the diverging colormap.
    """
    import matplotlib.colors as mcolors

    all_layers = [r["layer"] for r in head_results]
    all_heads  = [r["head"]  for r in head_results]
    n_layers   = max(all_layers)   # indices are 1-based; grid rows are 0-based
    n_heads    = max(all_heads)

    grid = np.full((n_layers, n_heads), np.nan)
    for r in head_results:
        grid[r["layer"] - 1, r["head"] - 1] = r[metric_key]

    cmap_hm = plt.cm.RdBu.copy()
    cmap_hm.set_bad(color="#dddddd")

    values = grid[~np.isnan(grid)]
    vmin = values.min()
    vmax = values.max()
    # ensure vcenter sits strictly between vmin and vmax
    vmin = min(vmin, vcenter - 1e-6)
    vmax = max(vmax, vcenter + 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.35 + 2), max(6, n_layers * 0.35 + 2)))
    im = ax.imshow(grid, aspect="auto", cmap=cmap_hm, norm=norm, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    top_heads = sorted(head_results, key=lambda r: r[metric_key])[:top_n_circled]
    for r in top_heads:
        ax.plot(r["head"] - 1, r["layer"] - 1, "o",
                mfc="none", mec="white", mew=1.2, ms=7, zorder=5)

    ax.set_xlabel("Head index", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(i + 1) for i in range(n_heads)], fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([str(i + 1) for i in range(n_layers)], fontsize=8)
    ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_causal_analysis(
    head_results: list[dict],
    baseline_delta: float,
    out_path: Path,
    top_n_circled: int = 64,
) -> None:
    """
    Saves two heatmap figures side by side in the same directory:
      - head_causal_delta.png         : raw knockout Δ per head
      - head_causal_delta_vs_base.png : Δ relative to baseline (Δ_ko − Δ_baseline)
    """
    out_path = Path(out_path)

    _plot_heatmap(
        head_results,
        metric_key="mean_delta",
        colorbar_label="Δ  =  avg logP(safe | ko) − avg logP(vuln | ko)",
        title=(
            f"Knockout Δ per head  (layer × head)\n"
            f"White = Δ = 0  |  blue = model prefers safe  |  red = model prefers vuln  |  "
            f"grey = not tested  |  white circles = top-{top_n_circled}"
        ),
        out_path=out_path,
        top_n_circled=top_n_circled,
        vcenter=0.0,
    )

    vs_base_path = out_path.parent / (out_path.stem + "_vs_base" + out_path.suffix)
    _plot_heatmap(
        head_results,
        metric_key="delta_vs_baseline",
        colorbar_label="Δ_ko − Δ_baseline  (negative = knockout reduced safe preference)",
        title=(
            f"Knockout Δ vs baseline per head  (layer × head)\n"
            f"White = no effect  |  red = head reduced safe preference (causally important)  |  "
            f"blue = head increased safe preference  |  "
            f"grey = not tested  |  white circles = top-{top_n_circled}"
        ),
        out_path=vs_base_path,
        top_n_circled=top_n_circled,
        vcenter=0.0,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def get_model_head_config(model) -> tuple[int, int]:
    cfg      = model.config
    n_heads  = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    return n_heads, head_dim


def run_analysis(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "head_causal_checkpoint.json"

    # ---- Load data ----
    print("Loading contrastive pairs ...")
    pairs = load_pairs(Path(args.pairs_file), args.cwe_id, args.max_pairs)
    if not pairs:
        raise ValueError(
            f"No pairs found for {args.cwe_id} in {args.pairs_file}. "
            "Check --cwe_id and --pairs_file."
        )
    print(f"  {len(pairs)} pairs loaded.")
    print("\n[Prompt sanity check — first pair]")
    print("  I_ori (first 200 chars):",
          pairs[0]["safe_messages"][0]["content"][:200].replace("\n", " "))

    print(f"\nLoading top-{args.top_k} heads from {args.head_results} ...")
    top_heads = load_top_heads(Path(args.head_results), args.top_k)
    print(f"  Top heads: {[(h['layer'], h['head']) for h in top_heads]}")

    # ---- Load model ----
    print(f"\nLoading model {args.model} ...")
    model, tokenizer = load_model(args.model, args.dtype, args.device)
    n_heads, head_dim = get_model_head_config(model)
    print(f"  n_heads={n_heads}, head_dim={head_dim}")

    # ---- Helper: run all pairs for one response side, optionally with a knockout hook ----
    def score_response(response_key: str, knocker: HeadKnockout | None = None) -> list[float]:
        """
        Teacher-forced avg log-prob of `response_key` tokens across all pairs.
        If knocker is provided, its hook is active during scoring.
        """
        all_lp: list[float] = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            encoding, n_prefix_list = prepare_batch(
                batch, response_key, tokenizer, args.device, model
            )
            resp_mask = build_response_mask(encoding["attention_mask"], n_prefix_list)

            if knocker is not None:
                knocker.response_mask = resp_mask  # (batch, seq) — per-sample masking

            lp = compute_avg_log_prob_batch(
                model,
                encoding["input_ids"],
                encoding["attention_mask"],
                resp_mask,
            )
            all_lp.extend(lp)

            if knocker is not None:
                knocker.response_mask = None

        return all_lp

    # ---- Resume from checkpoint if available ----
    head_results: list[dict] = []
    baseline_delta: float | None = None
    completed_ranks: set[int] = set()

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        baseline_delta = ckpt["baseline_delta"]
        head_results   = ckpt["head_results"]
        completed_ranks = {r["probe_rank"] for r in head_results}
        print(f"\nResuming from checkpoint: {len(completed_ranks)}/{args.top_k} heads done, "
              f"baseline_delta={baseline_delta:+.4f}")

    # ---- Baseline (no intervention) ----
    if baseline_delta is None:
        print("\nComputing baseline log-probs (no intervention) ...")
        base_safe_lp = score_response("safe_response")
        base_vuln_lp = score_response("vuln_response")

        base_deltas = [
            s - v
            for s, v in zip(base_safe_lp, base_vuln_lp)
            if not (math.isnan(s) or math.isnan(v))
        ]
        baseline_delta = float(np.mean(base_deltas)) if base_deltas else float("nan")
        print(f"  Baseline Delta = (1/N_safe)·Σ logP(r_safe_n|I_ori) - (1/N_vuln)·Σ logP(r_vuln_n|I_ori) = {baseline_delta:+.4f}")
        print(f"  (positive → model already prefers safe code under benign prompt)")
        with open(checkpoint_path, "w") as f:
            json.dump({"baseline_delta": baseline_delta, "head_results": []}, f)

    # ---- Per-head knockout ----
    for probe_rank, head_info in enumerate(top_heads):
        if probe_rank in completed_ranks:
            continue
        layer_idx = head_info["layer"]
        head_idx  = head_info["head"]
        probe_acc = head_info["val_accuracy"]
        label     = f"L{layer_idx:02d}H{head_idx:02d}"

        print(f"[{probe_rank+1}/{len(top_heads)}] Knocking out head {label} "
              f"(probe acc={probe_acc:.3f}) ...")

        knocker = HeadKnockout(model, layer_idx, head_idx, n_heads, head_dim)

        ko_safe_lp = score_response("safe_response", knocker=knocker)
        ko_vuln_lp = score_response("vuln_response", knocker=knocker)

        knocker.remove()

        # Delta per pair = (1/N_safe)·Σ logP(r_safe_n | I_ori, ko) - (1/N_vuln)·Σ logP(r_vuln_n | I_ori, ko)
        deltas: list[float] = []
        for s, v in zip(ko_safe_lp, ko_vuln_lp):
            if math.isnan(s) or math.isnan(v):
                continue
            deltas.append(s - v)

        mean_delta = float(np.mean(deltas)) if deltas else float("nan")
        std_delta  = float(np.std(deltas))  if deltas else float("nan")
        print(f"    Delta (ko) = {mean_delta:+.4f}  std = {std_delta:.4f}  "
              f"change vs baseline = {mean_delta - baseline_delta:+.4f}")

        head_results.append({
            "layer":            layer_idx,
            "head":             head_idx,
            "probe_rank":       probe_rank,
            "probe_accuracy":   probe_acc,
            "mean_delta":       mean_delta,
            "std_delta":        std_delta,
            "delta_vs_baseline": mean_delta - baseline_delta,
            "n_pairs":          len(deltas),
            "per_pair_deltas":  deltas,
        })
        with open(checkpoint_path, "w") as f:
            json.dump({"baseline_delta": baseline_delta, "head_results": head_results}, f)

    # Causal rank: 0 = most negative mean_delta (strongest dropout effect)
    sorted_by_delta = sorted(head_results, key=lambda x: x["mean_delta"])
    causal_rank_map = {(r["layer"], r["head"]): i for i, r in enumerate(sorted_by_delta)}
    for r in head_results:
        r["causal_rank"] = causal_rank_map[(r["layer"], r["head"])]

    heads_for_save = sorted(head_results, key=lambda x: x["causal_rank"])

    # ---- Save results ----
    results_path = output_dir / "head_causal_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "cwe_id":              args.cwe_id,
                "model":               args.model,
                "intervention":        "knockout",
                "n_pairs":             len(pairs),
                "top_k":               args.top_k,
                "prompt_construction": "rebuilt_safe_only",
                "baseline_delta":      baseline_delta,
                "heads":               heads_for_save,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results: {results_path}")
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ---- Plot ----
    plot_path = output_dir / "head_causal_delta.png"
    plot_causal_analysis(head_results, baseline_delta, plot_path)

    # ---- Summary ----
    print("\n--- Summary (sorted by causal rank, most negative Delta first) ---")
    print(f"  Baseline Delta = {baseline_delta:+.4f}")
    print(f"  {'Head':<10}  {'CausalRk':>8}  {'ProbeRk':>7}  {'ProbeAcc':>9}"
          f"  {'Delta(ko)':>10}  {'vs Base':>8}  {'Std':>7}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*7}")
    for r in heads_for_save:
        print(
            f"  L{r['layer']:02d}H{r['head']:02d}    "
            f"  {r['causal_rank']:>8d}"
            f"  {r['probe_rank']:>7d}"
            f"  {r['probe_accuracy']:>9.3f}"
            f"  {r['mean_delta']:>+10.4f}"
            f"  {r['delta_vs_baseline']:>+8.4f}"
            f"  {r['std_delta']:>7.4f}"
        )

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Causal attention head knockout analysis.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pairs_file", required=True,
        help="Path to contrastive pairs JSONL (cross_group recommended)",
    )
    parser.add_argument(
        "--head_results", required=True,
        help="Path to head_accuracy_results.json from train_probe.py",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--cwe_id", required=True,
        help="CWE to filter pairs on (e.g. cwe-089)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--top_k", type=int, default=16,
        help="Number of top probe-accurate heads to analyse (default: 16)",
    )
    parser.add_argument(
        "--max_pairs", type=int, default=None,
        help="Cap on number of pairs to process (default: all)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for model forward passes (default: 4)",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device: cuda / cpu / auto (default: cuda)",
    )
    _default_cwe_db = (
        Path(__file__).resolve().parents[1] / "data" / "cwe_official" / "all_cwe.json"
    )
    parser.add_argument(
        "--cwe_db", default=str(_default_cwe_db),
        help="Path to all_cwe.json (reserved for future use)",
    )
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
