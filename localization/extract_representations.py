"""
Extract LLM internal representations from contrastive pairs for a specific CWE.

Input format:
  Flat pair records with `question`, `safe_code`, `vuln_code`. The pair files
  store code only; user prompts are constructed at runtime from `question`
  using the templates in common/prompts.py. --prompt_mode selects the
  vulnerable side's prompt: `vanilla` (benign template, default) or
  `vuln_elicit` (the generic vulnerability-eliciting template, matching how
  the cross-prompt vulnerable generations were sampled). The safe side always
  uses the benign template. Each side is formatted as a proper chat completion
  (user + assistant) using the model's chat template, so the model sees
  exactly what it was trained on. Representations are extracted from
  assistant response tokens only.

Extraction modes:
  layer  — hidden state at each transformer layer  (n_layers+1 total)
  head   — per-head output before o_proj at each layer
  both   — layer + head (default)

Token aggregation (over response tokens only):
  response_last  — hidden state at the last response token (default)
  response_mean  — mean-pool over all response tokens

Output layout  {output_dir}/{model_slug}/{dataset_type}/{cwe_id}/{token_agg}/
  Layer files (one per layer):
    layer_{i:02d}.pt          → {"safe": Tensor(n_pairs, hidden_dim),
                                  "vuln": Tensor(n_pairs, hidden_dim)}

  Head files (one per layer × head, only when mode includes head):
    head_layer_{i:02d}_head_{j:02d}.pt
                              → {"safe": Tensor(n_pairs, head_dim),
                                 "vuln": Tensor(n_pairs, head_dim)}

  metadata.json               → pair IDs, run config, model info

Tensors are saved in the model's compute dtype (bfloat16 by default) —
full model precision, half the disk footprint of float32.

Usage:
  # Quick test (small model, 20 pairs)
  python extract_representations.py \\
      --input_file data/contrastive_pairs/llama31-8b_intra.jsonl \\
      --cwe_id cwe-022 \\
      --model meta-llama/Llama-3.2-1B-Instruct \\
      --max_pairs 20

  # Full run with target model
  python extract_representations.py \\
      --input_file data/contrastive_pairs/llama31-8b_intra.jsonl \\
      --cwe_id cwe-022 \\
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
      --dtype bfloat16 --device auto

Note: full sequences are used without truncation to preserve complete responses.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from common.prompts import (
    CODE_GENERATION_PROMPT,
    CODE_GENERATION_PROMPT_WITH_VULNERABILITY_GENERIC,
)


# --------------------------------------------------------------------------- #
# Chat formatting
# --------------------------------------------------------------------------- #

def pair_side(rec: dict, side: str, prompt_mode: str = "vanilla") -> dict:
    """
    Build one side of a flat pair record as {"prompt", "code"}.

    Prompts are constructed at runtime from the pair's question; the dataset
    stores code only. prompt_mode controls the vulnerable side's user prompt:
      vanilla     — benign generation template for both sides
      vuln_elicit — vulnerability-eliciting (generic) template for the
                    vulnerable side; the safe side always stays benign
    """
    if side == "vuln" and prompt_mode == "vuln_elicit":
        prompt = CODE_GENERATION_PROMPT_WITH_VULNERABILITY_GENERIC.format(
            question=rec["question"])
    else:
        prompt = CODE_GENERATION_PROMPT.format(question=rec["question"])
    return {"prompt": prompt, "code": rec[f"{side}_code"]}


def build_chat_inputs(
    entry: dict,
    tokenizer,
) -> tuple[list[int], int]:
    """
    Tokenize one pair side as a full chat (user turn + assistant response).
    The full sequence is kept without any truncation.

    Returns:
        input_ids  — complete token ids for the conversation
        n_prefix   — number of prefix tokens (system + user turns + assistant
                     header) that precede the actual response content
    """
    messages = [{"role": "user", "content": entry["prompt"]}]
    response = entry["code"]       # raw model output (may include ``` blocks)

    # Prefix: everything up to where the assistant starts generating
    prefix_ids: list[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    # Full conversation: user + assistant response
    full_messages = messages + [{"role": "assistant", "content": response}]
    full_ids: list[int] = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    n_prefix = min(len(prefix_ids), len(full_ids))
    return full_ids, n_prefix


def prepare_batch(
    entries: list[dict],
    tokenizer,
    device: str,
    model,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """
    Tokenize a batch with manual left-padding (so the last position is always
    the last real token, making response_last trivial to index).
    Full sequences are used — no truncation.

    Returns:
        encoding       — {"input_ids", "attention_mask"} tensors on device
        n_prefix_list  — per-sample prefix lengths
    """
    all_ids, n_prefix_list = [], []
    for entry in entries:
        ids, n_prefix = build_chat_inputs(entry, tokenizer)
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

    if device != "cpu":
        # For device_map="auto", find where embed_tokens actually lives
        tgt = next(model.parameters()).device
        input_ids      = input_ids.to(tgt)
        attention_mask = attention_mask.to(tgt)

    return {"input_ids": input_ids, "attention_mask": attention_mask}, n_prefix_list


def build_response_mask(
    attention_mask: torch.Tensor,  # (batch, seq) on any device
    n_prefix_list: list[int],
) -> torch.Tensor:                 # (batch, seq) bool on CPU
    """
    True for positions that belong to the assistant response.

    With left-padding:
      [ PAD…PAD | prefix tokens | response tokens ]
      response starts at: seq_len - real_len + n_prefix
    """
    cpu_mask = attention_mask.cpu()
    batch, seq_len = cpu_mask.shape
    response_mask = torch.zeros(batch, seq_len, dtype=torch.bool)

    for i, n_prefix in enumerate(n_prefix_list):
        real_len  = int(cpu_mask[i].sum().item())
        pad_len   = seq_len - real_len
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

    model_kwargs = {"torch_dtype": dtype, "output_hidden_states": True}
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    if device not in ("auto", "cpu") and not hasattr(model, "hf_device_map"):
        model = model.to(device)

    return model, tokenizer


# --------------------------------------------------------------------------- #
# Attention head hook
# --------------------------------------------------------------------------- #

class HeadOutputCapture:
    """
    Hooks the forward pre-pass of each attention o_proj to capture the
    concatenated per-head outputs before projection.
    Captured shape per layer: (batch, seq, n_heads * head_dim).
    Compatible with LLaMA / Mistral / Gemma (any model with an 'o_proj').
    """

    def __init__(self, model):
        self.captures: list[torch.Tensor] = []
        self._hooks: list = []
        for module in model.modules():
            if hasattr(module, "o_proj"):
                self._hooks.append(
                    module.o_proj.register_forward_pre_hook(self._hook_fn)
                )

    def _hook_fn(self, module, inputs):
        self.captures.append(inputs[0].detach().cpu())

    def clear(self):
        self.captures.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def agg_over_response(
    tensor: torch.Tensor,          # (batch, seq, dim) — any dtype
    response_mask: torch.Tensor,   # (batch, seq) bool, CPU
    token_agg: str,
    save_dtype: torch.dtype,
) -> torch.Tensor:                 # (batch, dim) in save_dtype
    """
    Aggregate tensor over response token positions.
    Computation is done in float32 for numerical stability, then cast
    to save_dtype so stored tensors match the model's precision.
    Falls back to last real token if no response tokens exist.
    """
    batch = tensor.shape[0]
    dim   = tensor.shape[-1]
    result = torch.zeros(batch, dim, dtype=torch.float32)
    t_f32  = tensor.cpu().float()   # promote for stable mean/indexing

    for i in range(batch):
        pos = response_mask[i].nonzero(as_tuple=True)[0]   # response positions
        if len(pos) == 0:
            result[i] = t_f32[i, -1]                       # fallback: last token
        elif token_agg == "response_last":
            result[i] = t_f32[i, pos[-1]]
        else:  # response_mean
            result[i] = t_f32[i, pos].mean(dim=0)

    return result.to(save_dtype)


# --------------------------------------------------------------------------- #
# Batch extraction — returns per-layer / per-head tensors in save_dtype
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_batch(
    model,
    tokenizer,
    entries: list[dict],
    mode: str,
    token_agg: str,
    device: str,
    head_capture: HeadOutputCapture | None,
    save_dtype: torch.dtype,
    n_heads: int,
    head_dim: int,
) -> dict:
    """
    Returns:
      result["layer"]: list of Tensor(batch, hidden_dim) — one per layer (n_layers+1)
      result["head"]:  list of list of Tensor(batch, head_dim) — [layer][head]
    """
    encoding, n_prefix_list = prepare_batch(entries, tokenizer, device, model)
    response_mask = build_response_mask(encoding["attention_mask"], n_prefix_list)

    if head_capture is not None:
        head_capture.clear()

    outputs = model(**encoding)
    hidden_states = outputs.hidden_states   # tuple of (batch, seq, hidden_dim)

    result = {}

    if mode in ("layer", "both"):
        result["layer"] = [
            agg_over_response(hs, response_mask, token_agg, save_dtype)
            for hs in hidden_states[1:]  # skip index 0 (embedding layer)
        ]

    if mode in ("head", "both") and head_capture is not None:
        # head_capture.captures: one tensor per layer, shape (batch, seq, n_heads*head_dim)
        layer_heads = []
        for cap in head_capture.captures:
            b, s, _ = cap.shape
            cap_split = cap.view(b, s, n_heads, head_dim)    # (b, seq, n_heads, head_dim)
            heads_for_layer = [
                agg_over_response(
                    cap_split[:, :, h, :],   # (b, seq, head_dim)
                    response_mask, token_agg, save_dtype,
                )
                for h in range(n_heads)
            ]
            layer_heads.append(heads_for_layer)
        result["head"] = layer_heads

    return result


# --------------------------------------------------------------------------- #
# Load pairs
# --------------------------------------------------------------------------- #

def _cwe_key(cwe_id: str) -> str:
    """Normalize a CWE id for comparison: 'cwe-022', 'CWE-22', '022', '22' -> '22'."""
    return str(cwe_id).lower().removeprefix("cwe-").lstrip("0") or "0"


def load_pairs(input_file: Path, cwe_id: str, max_pairs: int | None) -> list[dict]:
    pairs = []
    target = _cwe_key(cwe_id)
    with open(input_file) as f:
        for line in f:
            pair = json.loads(line)
            if _cwe_key(pair.get("cwe_id", "")) == target:
                pairs.append(pair)
                if max_pairs and len(pairs) >= max_pairs:
                    break
    return pairs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def slugify(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def main(args):
    input_file  = Path(args.input_file)
    dataset_type = input_file.stem.replace("contrastive_pairs_", "")

    print(f"Loading pairs for {args.cwe_id} from {input_file.name} ...")
    pairs = load_pairs(input_file, args.cwe_id, args.max_pairs)
    print(f"  {len(pairs)} pairs found")
    if not pairs:
        print("No pairs found for this CWE. Exiting.")
        return

    out_dir = (
        Path(args.output_dir)
        / slugify(args.model)
        / dataset_type
        / args.cwe_id
        / args.token_agg
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    print(f"Loading model: {args.model}  (dtype={args.dtype}, device={args.device}) ...")
    model, tokenizer = load_model(args.model, args.dtype, args.device)

    dtype_map   = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    save_dtype  = dtype_map[args.dtype]

    cfg     = model.config
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads

    head_capture = None
    if args.mode in ("head", "both"):
        head_capture = HeadOutputCapture(model)
        print(f"  Hooked {len(head_capture._hooks)} attention layers  "
              f"({n_heads} heads × {head_dim}-dim)")

    # Chat template sanity check on first pair
    first = pair_side(pairs[0], "safe", args.prompt_mode)
    ids, n_prefix = build_chat_inputs(first, tokenizer)
    print(f"\n  Chat template check (first pair, safe):")
    print(f"    total={len(ids)} tokens,  prefix={n_prefix},  response={len(ids)-n_prefix}")
    print(f"    prefix end : {tokenizer.decode(ids[n_prefix-5:n_prefix], skip_special_tokens=False)!r}")
    print(f"    response[0]: {tokenizer.decode(ids[n_prefix:n_prefix+8], skip_special_tokens=False)!r}")

    # Accumulators: layer_safe[layer_idx] = list of (batch, hidden_dim) tensors
    layer_safe: defaultdict[int, list] = defaultdict(list)
    layer_vuln: defaultdict[int, list] = defaultdict(list)
    # head_safe[layer_idx][head_idx] = list of (batch, head_dim) tensors
    head_safe:  defaultdict[int, defaultdict] = defaultdict(lambda: defaultdict(list))
    head_vuln:  defaultdict[int, defaultdict] = defaultdict(lambda: defaultdict(list))

    n_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
    print(f"\nExtracting ({n_batches} batches, batch_size={args.batch_size}) ...")

    for i in range(0, len(pairs), args.batch_size):
        bn = i // args.batch_size + 1
        batch = pairs[i : i + args.batch_size]

        print(f"  Batch {bn}/{n_batches}: safe ...", end="\r")
        safe_out = extract_batch(
            model, tokenizer, [pair_side(p, "safe", args.prompt_mode) for p in batch],
            args.mode, args.token_agg,
            args.device, head_capture, save_dtype, n_heads, head_dim,
        )

        print(f"  Batch {bn}/{n_batches}: vuln ...", end="\r")
        vuln_out = extract_batch(
            model, tokenizer, [pair_side(p, "vuln") for p in batch],
            args.mode, args.token_agg,
            args.device, head_capture, save_dtype, n_heads, head_dim,
        )

        if "layer" in safe_out:
            for l, (s, v) in enumerate(zip(safe_out["layer"], vuln_out["layer"])):
                layer_safe[l].append(s)
                layer_vuln[l].append(v)

        if "head" in safe_out:
            for l, (s_heads, v_heads) in enumerate(zip(safe_out["head"], vuln_out["head"])):
                for h, (s, v) in enumerate(zip(s_heads, v_heads)):
                    head_safe[l][h].append(s)
                    head_vuln[l][h].append(v)

    print(f"\nSaving .pt files to {out_dir} ...")

    # --- layer files ---
    if layer_safe:
        n_layers = len(layer_safe)
        hidden_dim = torch.cat(layer_safe[0], dim=0).shape[-1]
        print(f"  Layers: {n_layers} files  shape=({len(pairs)}, {hidden_dim})  dtype={save_dtype}")
        for l in range(n_layers):
            safe_tensor = torch.cat(layer_safe[l], dim=0)   # (n_pairs, hidden_dim)
            vuln_tensor = torch.cat(layer_vuln[l], dim=0)
            fname = out_dir / f"layer_{l+1:02d}.pt"
            torch.save({"safe": safe_tensor, "vuln": vuln_tensor}, fname)
        print(f"  Saved layer_01.pt … layer_{n_layers:02d}.pt")

    # --- head files ---
    if head_safe:
        n_layers_h = len(head_safe)
        print(f"  Heads: {n_layers_h * n_heads} files  "
              f"shape=({len(pairs)}, {head_dim})  dtype={save_dtype}")
        for l in range(n_layers_h):
            for h in range(n_heads):
                safe_tensor = torch.cat(head_safe[l][h], dim=0)   # (n_pairs, head_dim)
                vuln_tensor = torch.cat(head_vuln[l][h], dim=0)
                fname = out_dir / f"head_layer_{l+1:02d}_head_{h+1:02d}.pt"
                torch.save({"safe": safe_tensor, "vuln": vuln_tensor}, fname)
        print(f"  Saved head_layer_01_head_01.pt … "
              f"head_layer_{n_layers_h:02d}_head_{n_heads:02d}.pt")

    # --- metadata ---
    metadata = {
        "model":        args.model,
        "cwe_id":       args.cwe_id,
        "dataset_type": dataset_type,
        "input_file":   str(input_file),
        "mode":         args.mode,
        "token_agg":    args.token_agg,
        "dtype":        args.dtype,
        "n_pairs":      len(pairs),
        "n_layers":     len(layer_safe) if layer_safe else len(head_safe),
        "n_heads":      n_heads,
        "head_dim":     head_dim,
        "hidden_dim":   cfg.hidden_size,
        "pairs": [
            {
                "index":            idx,
                "id":               p["id"],
                "src_id":           p.get("src_id"),
                "source":           p.get("source"),
                "codeql_detections": p.get("vuln_codeql_detections", []),
            }
            for idx, p in enumerate(pairs)
        ],
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata → {meta_path}")

    if head_capture is not None:
        head_capture.remove()

    # Disk usage summary
    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.pt"))
    print(f"\nTotal .pt disk usage: {total_bytes / 1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract LLM representations from contrastive pairs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input_file", required=True,
                        help="Contrastive pairs JSONL (safe_only or cross_group)")
    parser.add_argument("--cwe_id", required=True,
                        help="CWE to filter on, e.g. cwe-089")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--output_dir", default="data/representations",
                        help="Root output directory")
    parser.add_argument("--mode", choices=["layer", "head", "both"], default="both",
                        help="What to extract")
    parser.add_argument("--prompt_mode", choices=["vanilla", "vuln_elicit"],
                        default="vanilla",
                        help="User prompt for the vulnerable side, built at runtime "
                             "from the question: benign template (vanilla) or the "
                             "generic vulnerability-eliciting template (vuln_elicit). "
                             "The safe side always uses the benign template.")
    parser.add_argument("--token_agg", choices=["response_last", "response_mean"],
                        default="response_last",
                        help=(
                            "response_last — last response token\n"
                            "response_mean — mean over all response tokens"
                        ))
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Samples per forward pass")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                        default="bfloat16",
                        help="Model + save dtype (bfloat16 = half disk, full model precision)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cuda | cpu | auto (multi-GPU)")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Process only first N pairs (testing)")
    args = parser.parse_args()
    main(args)
