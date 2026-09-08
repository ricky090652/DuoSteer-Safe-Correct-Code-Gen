"""
Steering evaluation via code generation.

Reads any task file (JSONL with `id` + `question`; the benign prompt is rendered at run time),
generates model responses under one or more intervention settings, and writes
results for downstream analysis (e.g. CodeQL).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETTINGS AND HOW THE STEERING WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  baseline
    No hooks.  Plain autoregressive generation; serves as the unsteered
    reference for every other condition.

  layer
    After every transformer layer forward pass, a fixed offset is added to
    the residual stream for every generated token:

        h_l  ←  h_l + α · v_l

    where h_l is the (batch, seq, hidden_dim) hidden state at layer l, α is
    the steering coefficient (--alpha), and v_l is the steering vector for
    that layer (loaded from --layer_steering_dir).

    With sigma-normalised vectors (--norm sigma in extract_steering_vector.py,
    now the default), v_l = sv_unit * proj_sigma, so α=1 shifts the
    residual-stream projection by exactly 1 standard deviation of the
    safe-vs-vulnerable projection distribution.  This makes α directly
    comparable across all layers, heads, and CWEs.

    The hook fires on the *output* of the layer's full sublayer stack (after
    the MLP), so the offset persists into every later layer and every later
    token.

  head
    After every attention sub-layer, a fixed offset is added to the
    pre-o_proj output slice belonging to each selected head:

        x[head_i]  ←  x[head_i] + α · v_{l,h}

    where x[head_i] is the slice (batch, seq, head_dim) cut from the
    concatenated head outputs (batch, seq, n_heads × head_dim), and v_{l,h}
    is the sigma-normalised steering vector for head h in layer l.
    With sigma normalisation α=1 ≡ 1σ shift, identical interpretation to
    the layer setting.

    top-k heads are selected by importance to safe code generation
    (see HEAD SELECTION below).  All selected heads share the same α.

  head_suppress
    Selects top_k heads by |delta_vs_baseline| and splits them 50/50 by
    causal direction.  No additive steering vector is used.

    Safe-promoting heads (delta_vs_baseline < 0, top top_k//2 by most
    negative delta): multiplicatively scaled by +scale_safe (> 1 amplifies):

        x[safe_head_i]  ←  x[safe_head_i] × scale_safe

    Vuln-promoting heads (delta_vs_baseline > 0, top top_k//2 by most
    positive delta): multiplicatively scaled by scale_vuln (< 0 inverts):

        x[vuln_head_j]  ←  x[vuln_head_j] × scale_vuln

    scale_vuln = -1.0 inverts direction; -2.0 inverts and amplifies;
    0.0 is a complete knockout.  Symmetric sweeping (scale_safe = s,
    scale_vuln = -s) simultaneously amplifies safety circuits and
    inverts vulnerability circuits.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER INDEX CONVENTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All layer and head indices throughout this script are 1-BASED, matching the
"layer" and "head" fields in probe accuracy and causal analysis result files.

  --layer_idx 1   →  model.model.layers[0]   (first transformer block)
  --layer_idx 32  →  model.model.layers[31]  (last block, 8B model)

The "layer" field in head_accuracy_results.json and head_causal_results.json
uses the same 1-based convention, so you can pass those values directly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEERING VECTOR METHOD (mean_diff vs probe)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

extract_steering_vector.py produces two families of sigma-normalised vectors
(--norm sigma, the only supported normalization):

  mean_diff   v = sv_unit * σ,  sv_unit = (mean(safe) − mean(vuln)) / ‖…‖
              Unsupervised; points from the vuln centroid toward the safe one.

  probe       v = sv_unit * σ,  sv_unit = −w / ‖w‖  (negated probe weight)
              Supervised; uses the trained binary classifier's decision normal.

  σ = std({ ⟨h_i, sv_unit⟩ })  over all training activations.
  With sigma normalisation, α=1 corresponds to a 1σ shift in the projection
  of the residual stream onto the safe direction — directly comparable across
  all layers, heads, and CWEs without any additional rescaling.

Both are supported.  Point --layer_steering_dir / --head_steering_dir at the
*method-specific leaf directory*, e.g.:

  data/steering_vectors/…/layer/mean_diff/    ← mean_diff layer vectors
  data/steering_vectors/…/layer/probe/        ← probe layer vectors
  data/steering_vectors/…/head/mean_diff/     ← mean_diff head vectors
  data/steering_vectors/…/head/probe/         ← probe head vectors

The method name is inferred from the directory path and included in the output
filename and steering_info so results from different methods don't collide.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEAD SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

--head_results accepts two file formats (auto-detected from content):

  head_accuracy_results.json  (from train_probe.py)
    Flat JSON list; each entry has "layer", "head", "val_accuracy".
    Safe importance = descending val_accuracy  (rank-0 = most discriminative).
    Vuln rank (reversed) = ascending val_accuracy (least discriminative for
    safe/vuln distinction; these heads are least informative about safety).

  head_causal_results.json  (from head_causal_analysis.py)
    JSON dict with key "heads"; each entry has "layer", "head", "causal_rank",
    "mean_delta", "delta_vs_baseline".
    Safe importance = ascending delta_vs_baseline (most negative first: knockout
    of these heads most hurts safe preference, so they contribute most to safe
    generation). Used as-is for H-MD-CS additive steering (top-k safe-promoting).
    H-Supp selects independently: top k//2 by most negative delta (safe-promoting)
    and top k//2 by most positive delta (vuln-promoting).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

One JSONL per setting under --output_dir.  Filenames encode key parameters:

  baseline.jsonl
  layer_L{layer:02d}_{method}_alpha{alpha:.1f}.jsonl
  head_top{k}_{method}_alpha{alpha:.1f}.jsonl
  head_suppress_top{k}_{method}_alpha{alpha:.1f}_vuln{k_vuln}x{scale_vuln}.jsonl

Each record keeps all input fields and adds:
  predicted_code   model's generated response (str)
  steering_info    dict with full intervention description

Interrupted runs resume automatically: items with an existing `predicted_code`
field are skipped.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Baseline
  python steer_eval.py \\
      --setting baseline \\
      --input_file data/prompt_code_gen_seccodeplt.jsonl \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --output_dir results/steering/cwe-022

  # Layer steering with mean_diff vector at layer 20 (1-based)
  python steer_eval.py \\
      --setting layer \\
      --input_file data/prompt_code_gen_seccodeplt.jsonl \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --layer_idx 20 \\
      --layer_steering_dir data/steering_vectors/.../layer/mean_diff \\
      --alpha 20 \\
      --output_dir results/steering/cwe-022

  # Head additive, top-5 by probe accuracy, probe vectors
  python steer_eval.py \\
      --setting head \\
      --input_file data/prompt_code_gen_seccodeplt.jsonl \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --head_results data/probes/.../head_accuracy_results.json \\
      --top_k 5 \\
      --head_steering_dir data/steering_vectors/.../head/probe \\
      --alpha 20 \\
      --output_dir results/steering/cwe-022

  # Head suppress: top-5 safe (additive) + bottom-3 vuln (knockout)
  python steer_eval.py \\
      --setting head_suppress \\
      --input_file data/prompt_code_gen_seccodeplt.jsonl \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --head_results data/causal_analysis/.../head_causal_results.json \\
      --top_k 5 --alpha 20 \\
      --head_steering_dir data/steering_vectors/.../head/mean_diff \\
      --k_vuln 3 --scale_vuln 0.0 \\
      --output_dir results/steering/cwe-022

  # All settings in one model-load pass
  python steer_eval.py \\
      --setting all \\
      --input_file data/prompt_code_gen_seccodeplt.jsonl \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --layer_idx 20 \\
      --layer_steering_dir data/steering_vectors/.../layer/mean_diff \\
      --head_steering_dir  data/steering_vectors/.../head/mean_diff \\
      --head_results data/probes/.../head_accuracy_results.json \\
      --alpha 20 --top_k 5 \\
      --k_vuln 3 --scale_vuln 0.0 \\
      --output_dir results/steering/cwe-022
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from prompts import CODE_GENERATION_PROMPT  # single prompt source, rendered at run time


# --------------------------------------------------------------------------- #
# Activation hooks
# --------------------------------------------------------------------------- #

class SteerLayer:
    """
    Adds α·v to the residual stream after a transformer layer.

    Hook fires on the layer *output*, so the offset accumulates into all
    subsequent layers and all subsequent generated tokens.

    layer_idx : int  1-based (layer 1 = first transformer block)
    """

    def __init__(self, model, layer_idx: int, steering_vec: torch.Tensor, alpha: float):
        self._sv     = steering_vec          # (hidden_dim,) float32, sigma-normed
        self._alpha  = alpha
        # Convert 1-based → 0-based for Python list indexing
        self._handle = model.model.layers[layer_idx - 1].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        offset = (self._alpha * self._sv).to(hidden.device, hidden.dtype)
        hidden = hidden + offset
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def remove(self):
        self._handle.remove()


class SteerHead:
    """
    Adds α·v to one attention head's output slice (before o_proj).

    The hook fires as a forward *pre*-hook on o_proj, receiving the
    concatenated head outputs (batch, seq, n_heads × head_dim).  It modifies
    only the slice belonging to the target head:

        x[:, :, head_idx*head_dim : (head_idx+1)*head_dim]  +=  α · v

    layer_idx, head_idx : int  both 1-based
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        head_idx: int,
        head_dim: int,
        steering_vec: torch.Tensor,   # (head_dim,) float32, sigma-normed
        alpha: float,
    ):
        self._head_idx = head_idx - 1   # convert to 0-based for slice arithmetic
        self._head_dim = head_dim
        self._sv       = steering_vec
        self._alpha    = alpha

        o_proj = _find_o_proj(model, layer_idx - 1)   # 0-based layer
        if o_proj is None:
            raise ValueError(f"Cannot find o_proj for layer {layer_idx}.")
        self._handle = o_proj.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x      = inputs[0]             # (batch, seq, n_heads * head_dim)
        offset = (self._alpha * self._sv).to(x.device, x.dtype)
        start  = self._head_idx * self._head_dim
        end    = start + self._head_dim
        x      = x.clone()
        x[:, :, start:end] += offset
        return (x,)

    def remove(self):
        self._handle.remove()


class SuppressHead:
    """
    Multiplies one attention head's output slice by a scale factor (before o_proj).

        x[:, :, head_idx*head_dim : (head_idx+1)*head_dim]  *=  scale_factor

    layer_idx, head_idx : int  both 1-based
    scale_factor > 1   →  amplify (safe-promoting heads in H-Supp)
    scale_factor ∈ (0,1) →  partial attenuation
    scale_factor = 0.0  →  complete knockout
    scale_factor < 0   →  invert direction (vuln-promoting heads in H-Supp)
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        head_idx: int,
        head_dim: int,
        scale_factor: float,
    ):
        self._head_idx   = head_idx - 1   # 0-based
        self._head_dim   = head_dim
        self._scale      = scale_factor

        o_proj = _find_o_proj(model, layer_idx - 1)
        if o_proj is None:
            raise ValueError(f"Cannot find o_proj for layer {layer_idx}.")
        self._handle = o_proj.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x     = inputs[0]
        start = self._head_idx * self._head_dim
        end   = start + self._head_dim
        x     = x.clone()
        x[:, :, start:end] *= self._scale
        return (x,)

    def remove(self):
        self._handle.remove()


def _find_o_proj(model, layer_idx: int):
    """Return the o_proj module at 0-based layer_idx, or None if not found."""
    o_projs = [m.o_proj for m in model.modules() if hasattr(m, "o_proj")]
    return o_projs[layer_idx] if layer_idx < len(o_projs) else None


# --------------------------------------------------------------------------- #
# Steering vector loading
# --------------------------------------------------------------------------- #

def _infer_sv_method(steering_dir: str) -> str:
    """
    Infer whether steering vectors are 'mean_diff' or 'probe' from the
    directory path.  Falls back to 'unknown' if neither is found.
    """
    for part in reversed(Path(steering_dir).parts):
        if part in ("mean_diff", "probe"):
            return part
    return "unknown"


def load_layer_sv(
    steering_dir: str, layer_idx: int,
) -> tuple[torch.Tensor, str, float]:
    """
    Load the steering vector for layer_idx (1-based) from steering_dir.

    Returns (vector, method, raw_norm).
    The vector is sigma-normalised (magnitude = proj_sigma), shape (hidden_dim,).
    alpha=1 ≡ 1σ shift along the safe direction; directly comparable across layers and heads.
    """
    path = Path(steering_dir) / f"steering_vector_layer_{layer_idx:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Layer steering vector not found: {path}")
    data     = torch.load(path, map_location="cpu", weights_only=True)
    method   = data.get("method", _infer_sv_method(steering_dir))
    raw_norm = float(data.get("raw_norm", 1.0))
    return data["steering_vector"].float(), method, raw_norm


def load_head_sv(
    steering_dir: str, layer_idx: int, head_idx: int,
) -> tuple[torch.Tensor, str, float]:
    """
    Load the steering vector for (layer_idx, head_idx), both 1-based.

    Returns (vector, method, raw_norm).  Shape (head_dim,), float32, sigma-normalised.
    alpha=1 ≡ 1σ shift along the safe direction; directly comparable across layers and heads.
    """
    path = Path(steering_dir) / f"steering_vector_head_{layer_idx:02d}_{head_idx:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Head steering vector not found: {path}")
    data     = torch.load(path, map_location="cpu", weights_only=True)
    method   = data.get("method", _infer_sv_method(steering_dir))
    raw_norm = float(data.get("raw_norm", 1.0))
    return data["steering_vector"].float(), method, raw_norm


# --------------------------------------------------------------------------- #
# Head results loading and ranking
# --------------------------------------------------------------------------- #

def load_head_results(path: str) -> tuple[list[dict], str]:
    """
    Load head importance results from either probe or causal analysis JSON.

    Returns
    -------
    ranked_safe : list[dict]
        Entries sorted by *safe* importance, best first.
        Each entry has at minimum "layer" and "head".
    results_type : "probe" | "causal"

    Probe format   (head_accuracy_results.json)
      Flat list with "val_accuracy".  Sorted descending → rank-0 is the head
      whose activations best separate safe from vulnerable code.

    Causal format  (head_causal_results.json)
      Dict with "heads" list and "causal_rank" per entry.  Sorted ascending
      by causal_rank → rank-0 is the head whose knockout causes the largest
      drop in Δ = mean_logP(safe) − mean_logP(vuln).
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "heads" in data:
        # Rank by delta_vs_baseline ascending: most negative first = most
        # safe-promoting (knockout of these heads most hurts safe preference).
        # Used by select_safe_heads for H-MD-CS additive steering.
        # select_suppress_heads re-sorts by absolute value internally for H-Supp.
        ranked_safe = sorted(
            data["heads"],
            key=lambda r: r.get("delta_vs_baseline", r.get("mean_delta", 0)),
        )
        return ranked_safe, "causal"
    else:
        ranked_safe = sorted(data, key=lambda r: r["val_accuracy"], reverse=True)
        return ranked_safe, "probe"


def select_safe_heads(ranked_safe: list[dict], k: int) -> list[dict]:
    """Top-k heads by safe importance (first k of ranked_safe)."""
    return ranked_safe[:k]


def select_suppress_heads(ranked_causal: list[dict], k: int) -> tuple[list[dict], list[dict]]:
    """
    Split k heads 50/50 into safe-promoting and vuln-promoting for H-Supp.

    Selects independently by direction: k//2 heads with the most negative
    delta_vs_baseline (safe-promoting) and k//2 with the most positive
    delta_vs_baseline (vuln-promoting). This picks the most impactful heads
    in each direction, regardless of how the other direction ranks.

    Returns (safe_heads, vuln_heads), each of length k//2.
    """
    half = k // 2
    by_delta = sorted(
        ranked_causal,
        key=lambda r: r.get("delta_vs_baseline", r.get("mean_delta", 0)),
    )
    safe_heads = by_delta[:half]              # most negative delta → safe-promoting
    vuln_heads = list(reversed(by_delta))[:half]  # most positive delta → vuln-promoting
    return safe_heads, vuln_heads


def _head_label(h: dict) -> str:
    return f"L{h['layer']:02d}H{h['head']:02d}"


def _head_summary(h: dict, results_type: str) -> dict:
    d = {"layer_idx": h["layer"], "head_idx": h["head"], "label": _head_label(h)}
    if results_type == "probe":
        d["val_accuracy"] = h.get("val_accuracy")
    else:
        d["causal_rank"] = h.get("causal_rank")
        d["mean_delta"]  = h.get("mean_delta")
    return d


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_name: str, dtype_str: str, device: str):
    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"torch_dtype": dtype_map[dtype_str]}
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    if device not in ("auto", "cpu") and not hasattr(model, "hf_device_map"):
        model = model.to(device)
    return model, tokenizer


def get_head_config(model) -> tuple[int, int]:
    cfg      = model.config
    n_heads  = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    return n_heads, head_dim


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #



def build_prompt(item: dict, tokenizer) -> str:
    """Return the fully-formatted prompt string for an input item."""
    if "messages" in item and item["messages"]:
        return tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
    question = item.get("question", "")
    messages = [{"role": "user", "content": CODE_GENERATION_PROMPT.format(question=question)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, args) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings,
    )
    input_ids      = inputs["input_ids"].to(next(model.parameters()).device)
    attention_mask = inputs["attention_mask"].to(input_ids.device)

    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else 1.0,
        top_p=args.top_p             if args.do_sample else 1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Output filename
# --------------------------------------------------------------------------- #

def make_output_filename(setting: str, args, sv_method: str = "unknown") -> str:
    if setting == "baseline":
        return "baseline.jsonl"
    if setting == "layer":
        return f"layer_L{args.layer_idx:02d}_{sv_method}_alpha{args.alpha:.1f}.jsonl"
    if setting == "head":
        return f"head_top{args.top_k}_{sv_method}_alpha{args.alpha:.1f}.jsonl"
    if setting == "head_suppress":
        half = args.top_k // 2
        return (
            f"head_suppress_top{args.top_k}"
            f"_safe{half}x{args.scale_safe:.1f}"
            f"_vuln{half}x{args.scale_vuln:.1f}.jsonl"
        )
    raise ValueError(f"Unknown setting: {setting!r}")


# --------------------------------------------------------------------------- #
# Per-setting execution
# --------------------------------------------------------------------------- #

def run_setting(
    setting: str,
    items: list[dict],
    model,
    tokenizer,
    args,
    output_dir: Path,
    head_dim: int,
) -> None:
    """
    Register hooks, generate responses for all pending items, write results
    incrementally, then always remove hooks (even on exception).
    """
    # ---- Resume support: discover already-completed items ----
    # Determine filename after a possible early peek at the steering vector
    # to get the method; for the baseline the name is fixed.
    sv_method = "unknown"
    if setting == "layer":
        _, sv_method, _ = load_layer_sv(args.layer_steering_dir, args.layer_idx)
    elif setting == "head":
        sv_method = _infer_sv_method(args.head_steering_dir)
    # head_suppress: no steering vector, sv_method not used in filename

    output_file = output_dir / make_output_filename(setting, args, sv_method)

    done_ids: set[str] = set()
    existing: list[dict] = []
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
            print(f"  resume: {len(done_ids)} items already done")

    pending = [it for it in items if it["id"] not in done_ids]
    print(f"  {len(pending)} items to generate  →  {output_file.name}")
    if not pending:
        return

    # ---- Build hooks and steering_info ----
    hooks: list = []
    steering_info: dict = {"setting": setting}

    if setting == "baseline":
        steering_info.update({"mode": None, "targets": []})

    elif setting == "layer":
        sv, sv_method, raw_norm = load_layer_sv(args.layer_steering_dir, args.layer_idx)
        hooks.append(SteerLayer(model, args.layer_idx, sv, args.alpha))
        steering_info.update({
            "mode":       "layer",
            "sv_method":  sv_method,
            "alpha":      args.alpha,
            "raw_norm":   raw_norm,
            "layer_idx":  args.layer_idx,
        })
        print(f"  L{args.layer_idx:02d}  α={args.alpha}  raw_norm={raw_norm:.4f}  sv={sv_method}")

    elif setting == "head":
        ranked_safe, rtype = load_head_results(args.head_results)
        targets = select_safe_heads(ranked_safe, args.top_k)
        raw_norms = []
        for t in targets:
            sv, sv_method, raw_norm = load_head_sv(args.head_steering_dir, t["layer"], t["head"])
            raw_norms.append(raw_norm)
            hooks.append(SteerHead(model, t["layer"], t["head"], head_dim, sv, args.alpha))
        mean_raw_norm = sum(raw_norms) / len(raw_norms) if raw_norms else 1.0
        steering_info.update({
            "mode":          "head",
            "sv_method":     sv_method,
            "results_type":  rtype,
            "alpha":         args.alpha,
            "mean_raw_norm": mean_raw_norm,
            "targets":       [_head_summary(t, rtype) for t in targets],
        })
        print(f"  heads=[{', '.join(_head_label(t) for t in targets)}]"
              f"  α={args.alpha}  mean_raw_norm={mean_raw_norm:.5f}"
              f"  sv={sv_method}  sel={rtype}")

    elif setting == "head_suppress":
        # 50/50 split of top-k heads by |delta_vs_baseline|.
        # Safe-promoting heads (delta < 0) scaled by +scale_safe (amplify).
        # Vuln-promoting heads (delta > 0) scaled by scale_vuln (invert, e.g. -2.0).
        # No additive steering vector used.
        ranked_causal, rtype = load_head_results(args.head_results)
        safe_targets, vuln_targets = select_suppress_heads(ranked_causal, args.top_k)

        for t in safe_targets:
            hooks.append(SuppressHead(model, t["layer"], t["head"], head_dim, args.scale_safe))
        for t in vuln_targets:
            hooks.append(SuppressHead(model, t["layer"], t["head"], head_dim, args.scale_vuln))

        steering_info.update({
            "mode":         "head_suppress",
            "results_type": rtype,
            "top_k":        args.top_k,
            "scale_safe":   args.scale_safe,
            "scale_vuln":   args.scale_vuln,
            "safe_targets": [_head_summary(t, rtype) for t in safe_targets],
            "vuln_targets": [_head_summary(t, rtype) for t in vuln_targets],
        })
        print(f"  safe=[{', '.join(_head_label(t) for t in safe_targets)}]  ×{args.scale_safe}")
        print(f"  vuln=[{', '.join(_head_label(t) for t in vuln_targets)}]  ×{args.scale_vuln}")

    # ---- Generate ----
    results = list(existing)
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
# Main
# --------------------------------------------------------------------------- #

_ALL_SETTINGS = ["baseline", "layer", "head", "head_suppress"]


def main(args):
    settings = _ALL_SETTINGS if "all" in args.setting else list(dict.fromkeys(args.setting))
    print(f"Settings: {settings}")

    # Build sweep lists — model is loaded once and all combinations run in sequence.
    alphas  = args.alpha_list if args.alpha_list else [args.alpha]
    layers  = args.layer_list if args.layer_list else ([args.layer_idx] if args.layer_idx is not None else [None])
    top_ks  = args.top_k_list if args.top_k_list else [args.top_k]

    for s in settings:
        if s == "layer":
            if not layers or layers[0] is None:
                raise ValueError("--layer_idx or --layer_list required for 'layer' setting.")
            if not args.layer_steering_dir:
                raise ValueError("--layer_steering_dir required for 'layer' setting.")
        if s in ("head", "head_suppress"):
            if not args.head_results:
                raise ValueError(f"--head_results required for '{s}' setting.")
        if s == "head":
            if not args.head_steering_dir:
                raise ValueError("--head_steering_dir required for 'head' setting.")
        if s == "head_suppress":
            min_k = min(top_ks)
            if min_k < 2:
                raise ValueError("--top_k / --top_k_list must be >= 2 for 'head_suppress'.")

    with open(args.input_file) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if not items:
        raise ValueError(f"No items found in {args.input_file!r}.")
    print(f"Input: {len(items)} items")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {args.model!r} ...")
    model, tokenizer = load_model(args.model, args.dtype, args.device)
    _, head_dim = get_head_config(model)
    print(f"  head_dim={head_dim}")

    for setting in settings:
        print(f"\n{'='*60}\nSetting: {setting}\n{'='*60}")

        if setting == "baseline":
            run_setting(setting, items, model, tokenizer, args, output_dir, head_dim)

        elif setting == "layer":
            for layer_idx in layers:
                for alpha in alphas:
                    a = copy.copy(args)
                    a.layer_idx = layer_idx
                    a.alpha     = alpha
                    print(f"\n-- layer {layer_idx}  α={alpha} --")
                    run_setting(setting, items, model, tokenizer, a, output_dir, head_dim)

        elif setting == "head":
            for top_k in top_ks:
                for alpha in alphas:
                    a = copy.copy(args)
                    a.top_k = top_k
                    a.alpha = alpha
                    print(f"\n-- top_k={top_k}  α={alpha} --")
                    run_setting(setting, items, model, tokenizer, a, output_dir, head_dim)

        elif setting == "head_suppress":
            # alpha treated as scale_safe; scale_vuln = -alpha (symmetric)
            for top_k in top_ks:
                for alpha in alphas:
                    a = copy.copy(args)
                    a.top_k       = top_k
                    a.scale_safe  = alpha
                    a.scale_vuln  = -alpha
                    print(f"\n-- top_k={top_k}  scale_safe={alpha}  scale_vuln={-alpha} --")
                    run_setting(setting, items, model, tokenizer, a, output_dir, head_dim)

    print(f"\nAll done.  Results in: {output_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steering evaluation via code generation on any prompt dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # I/O
    parser.add_argument("--input_file", required=True,
        help="Input JSONL.  Each record must have 'id' and either 'messages'\n"
             "(chat-template list) or 'question' (plain text).")
    parser.add_argument("--output_dir", required=True,
        help="Directory to write per-setting JSONL output files.")

    # Settings
    parser.add_argument("--setting", nargs="+", default=["all"],
        choices=["all", "baseline", "layer", "head", "head_suppress"],
        help=(
            "Intervention setting(s) to run (default: all).\n"
            "  baseline       no steering\n"
            "  layer          additive offset to residual stream after layer l\n"
            "                   h_l ← h_l + α·v_l\n"
            "  head           additive offset to pre-o_proj slice of top-k heads\n"
            "                   x[h] ← x[h] + α·v_{l,h}\n"
            "  head_suppress  same as 'head' for safe heads, plus multiplicative\n"
            "                 suppression of the k_vuln most vuln-correlated heads\n"
            "                   x[j] ← x[j] × scale_vuln"
        ))

    # Model
    parser.add_argument("--model", required=True,
        help="HuggingFace model name or local path.")
    parser.add_argument("--dtype", default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16).")
    parser.add_argument("--device", default="cuda",
        help="Device: cuda / cpu / auto (default: cuda).")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=2048,
        help="Max tokens to generate per item (default: 2048).")
    parser.add_argument("--do_sample", action="store_true",
        help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0,
        help="Sampling temperature (only with --do_sample, default: 1.0).")
    parser.add_argument("--top_p", type=float, default=1.0,
        help="Nucleus sampling p (only with --do_sample, default: 1.0).")

    # Layer setting
    parser.add_argument("--layer_idx", type=int, default=None,
        help=(
            "Layer to steer — 1-BASED, matching the 'layer' field in probe/causal\n"
            "results.  layer_idx=1 is the first transformer block.\n"
            "Required for 'layer' setting."
        ))
    parser.add_argument("--layer_steering_dir", default=None,
        help=(
            "Leaf directory with layer steering vectors.\n"
            "Must contain: steering_vector_layer_{N:02d}.pt\n"
            "Point to the method-specific subdirectory, e.g.:\n"
            "  data/steering_vectors/.../layer/mean_diff\n"
            "  data/steering_vectors/.../layer/probe\n"
            "Required for 'layer' setting."
        ))

    # Head settings
    parser.add_argument("--head_steering_dir", default=None,
        help=(
            "Leaf directory with per-head steering vectors.\n"
            "Must contain: steering_vector_head_{L:02d}_{H:02d}.pt\n"
            "Point to the method-specific subdirectory, e.g.:\n"
            "  data/steering_vectors/.../head/mean_diff\n"
            "  data/steering_vectors/.../head/probe\n"
            "Required for 'head' and 'head_suppress' settings."
        ))
    parser.add_argument("--head_results", default=None,
        help=(
            "Path to head importance results JSON (auto-detected format):\n"
            "  head_accuracy_results.json (train_probe.py)\n"
            "    Flat list with 'val_accuracy'; safe rank = descending accuracy.\n"
            "  head_causal_results.json (head_causal_analysis.py)\n"
            "    Dict with 'heads' list and 'causal_rank'; safe rank = ascending\n"
            "    causal_rank (0 = most causally important for safe generation).\n"
            "Required for 'head' and 'head_suppress' settings."
        ))

    # Shared: additive coefficient and safe-head count
    parser.add_argument("--alpha", type=float, default=3.0,
        help="Additive steering coefficient α (default: 20.0). Used when --alpha_list\n"
             "is not set. For 'head_suppress', alpha is treated as scale_safe and\n"
             "scale_vuln is set to -alpha (symmetric).")
    parser.add_argument("--alpha_list", type=float, nargs="+", default=None,
        help="Sweep over multiple α values in a single model-load pass.\n"
             "Overrides --alpha when set. Example: --alpha_list 1 1.5 2 2.5 3 5 8")
    parser.add_argument("--layer_list", type=int, nargs="+", default=None,
        help="Sweep over multiple layer indices for 'layer' setting (1-based).\n"
             "Overrides --layer_idx when set. Example: --layer_list $(seq 1 32)")
    parser.add_argument("--top_k_list", type=int, nargs="+", default=None,
        help="Sweep over multiple top_k values for 'head'/'head_suppress' settings.\n"
             "Overrides --top_k when set. Example: --top_k_list 5 10 16")
    parser.add_argument("--top_k", type=int, default=1,
        help="Number of top safe-important heads to steer additively\n"
             "('head' and 'head_suppress', default: 1).")

    # head_suppress only
    parser.add_argument("--scale_safe", type=float, default=2.0,
        help="Scale factor applied to safe-promoting heads in 'head_suppress'\n"
             "(default: 2.0).  > 1 amplifies; 1 = identity; (0,1) attenuates.\n"
             "Top top_k//2 heads by most negative delta_vs_baseline are selected.")
    parser.add_argument("--scale_vuln", type=float, default=-2.0,
        help="Scale factor applied to vuln-promoting heads in 'head_suppress'\n"
             "(default: -2.0).  Negative = invert direction; 0.0 = knockout.\n"
             "Top top_k//2 heads by most positive delta_vs_baseline are selected.")

    args = parser.parse_args()
    main(args)
