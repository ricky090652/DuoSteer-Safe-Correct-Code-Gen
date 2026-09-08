# Stage 2 — Localization: Probing and Causal Head Knockout

Identifies where the safe-vs-vulnerable distinction is linearly encoded (probing) and which attention heads causally drive it (knockout). Requires the Stage 1 contrastive pairs.

## 1. Extract representations

Per-layer residual streams and per-head outputs (before `o_proj`), mean-pooled over assistant-response tokens. The released `_intra` files are the probe sets:

```bash
python localization/extract_representations.py \
    --input_file data/contrastive_pairs/llama31-8b_intra.jsonl \
    --cwe_id cwe-022 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output_dir data/representations \
    --mode both --token_agg response_mean \
    --dtype bfloat16 --device auto
```

Repeat per CWE. Probes and steering vectors use the `_intra` files (probing draws a question-level 80/20 split; the same 80% side trains the steering vectors); the `_cross` files are reserved for causal patching. Prompts are built at runtime from each pair's `question`: the default `--prompt_mode vanilla` uses the benign template for both sides; pass `--prompt_mode vuln_elicit` to give the vulnerable side the generic vulnerability-eliciting template (matching how the `_cross` vulnerable generations were sampled). Outputs one `.pt` file per layer and per head, plus `metadata.json`.

## 2. Train linear probes

```bash
python localization/train_probe.py \
    --rep_dir data/representations/meta-llama_Meta-Llama-3_1-8B-Instruct/llama31-8b_intra/cwe-022/response_mean \
    --mode layer --output_dir data/probes/cwe-022

python localization/train_probe.py --rep_dir ... --mode head --output_dir data/probes/cwe-022
```

Logistic-regression probes per layer or per (layer, head) with a question-level 80/20 train/validation split. Head mode writes `plots/head_accuracy_results.json`, which ranks heads by probe accuracy for the next step.

## 3. Causal head knockout

Zero each head's output at all response positions and measure the change in the model's preference for the safe over the vulnerable continuation under teacher forcing on the benign prompt, using the released `_cross` files:

```bash
python localization/head_causal_analysis.py \
    --pairs_file data/contrastive_pairs/llama31-8b_cross.jsonl \
    --head_results data/probes/cwe-022/plots/head_accuracy_results.json \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --cwe_id cwe-022 \
    --output_dir data/causal_analysis/cwe-022 \
    --top_k 256
```

`--top_k 256` evaluates the top-256 probe-ranked heads (the default setting). The output `head_causal_results.json` is the causal ranking consumed by the steering stage; heads with negative delta are safe-promoting. The run checkpoints and resumes automatically.

## Other models

All commands above take the model as an argument; replicate on another model by swapping `--model` (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct`, which has 28 layers and 28 heads) and keeping the same pair files. For the cross-prompt representations our Qwen run used `--token_agg response_last`. The correctness contrast reuses `data/correctness_pairs/llama/{cwe}_train.jsonl` for any model.
