# Stage 3 — Steering and DuoSteer

Builds steering vectors from the Stage 2 representations, runs single-vector steering, constructs the correctness direction, and runs DuoSteer (simultaneous safety + correctness injection at causally identified head sets).

## 1. Safety steering vectors

```bash
python steering/extract_steering_vector.py \
    --representations_dir data/representations \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --cwe_id cwe-022 \
    --dataset_type llama31-8b_intra \
    --token_agg response_mean \
    --mode both --method both --norm sigma \
    --output_dir data/steering_vectors --probe_dir data/probes/cwe-022
```

`--dataset_type` names the representation slug from Stage 2 (it equals the pair file's basename, e.g. `llama31-8b_intra`). `--method mean_diff` builds the mean-difference (MD) vector; `--method probe` uses the probe weight direction (PD); `--norm sigma` scales the unit direction by the standard deviation of the activations' projections onto it, so steering strength `alpha` moves activations by about `alpha` standard deviations along the direction.

## 2. Single-vector steering on the evaluation set

```bash
python steering/steer_eval.py \
    --input_file data/eval_tasks/seccodeplt_cwe022.jsonl \
    --output_dir results/steering_eval/cwe-022 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --setting head \
    --head_steering_dir data/steering_vectors/<model_slug>/<dataset_type>/cwe-022/response_mean/head/mean_diff \
    --head_results data/causal_analysis/cwe-022/head_causal_results.json \
    --alpha_list 1 2 3 5 10 --top_k_list 16 32 64 128
```

`--setting baseline` produces the unsteered condition; `--setting layer` steers the full residual stream at a chosen layer. Selecting heads with the probe ranking instead of the causal ranking gives the probe-selected variant. `steering/steer_experiment.py` is the log-probability-space diagnostic (measures the safe-vs-vulnerable margin as a function of `alpha` without generation).

## 3. Correctness direction (DuoSteer data pipeline)

The correctness vector is estimated on the distribution where it will act: safety-steered outputs that are CodeQL-safe, contrasting functionally correct vs incorrect ones.

```bash
# 3a. Training prompts per CWE (one task per question, disjoint from the evaluation set)
#     -> data/double_steering/tasks/train_tasks_cwe-022.jsonl, ...
#     --pair_prefix qwen25-coder-7b selects another model's pairs; --cross_for_all
#     also draws cross-pair questions for CWE-022/079 (small Qwen intra pools)
python steering/prepare_correctness_prompts.py

# 3b. Safety-only steered outputs on those prompts, one output directory per
#     steering configuration: data/double_steering/steered_raw/<cwe>/<config>/
python steering/steer_eval.py \
    --input_file data/double_steering/tasks/train_tasks_cwe-022.jsonl \
    --output_dir data/double_steering/steered_raw/cwe-022/causal_md \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --setting head \
    --head_steering_dir data/steering_vectors/<model_slug>/<dataset_type>/cwe-022/response_mean/head/mean_diff \
    --head_results data/causal_analysis/cwe-022/head_causal_results.json \
    --alpha_list 1 2 3 5 --top_k_list 16 32 64

# 3c. Quality filter (code fence, length, ast.parse) over every <cwe>/<config>/*.jsonl
python steering/filter_steered.py

# 3d. Keep only CodeQL-safe outputs (--base_dir moves the whole tree, e.g. data/double_steering/qwen)
python steering/codeql_filter.py --cwe cwe-022   # repeat per CWE

# 3e. GPT-4.1 correctness labels via the OpenAI Batch API
export OPENAI_API_KEY=...
python steering/correctness_batch_prepare.py
python steering/correctness_batch_submit.py
python steering/correctness_batch_collect.py

# 3f. Build (safe-and-correct, safe-but-incorrect) pairs, grouped by question
#     (--pair_target caps every CWE, e.g. 300 for Qwen)
python steering/build_correctness_pairs.py
```

The released `data/correctness_pairs/llama/` files are the output of 3f. Then extract representations and vectors for the correctness contrast (same commands as Stages 2.1 and 3.1, pointed at the correctness pairs), and run `localization/head_causal_analysis.py` on them to obtain the correctness-causal head ranking.

Optionally orthogonalize the correctness vectors against the safety vectors (used by DuoSteer method `A`; method `E` uses the raw vectors):

```bash
python steering/orthogonalize_vectors.py
```

## 4. DuoSteer

```bash
python steering/duosteer_eval.py \
    --method A E \
    --cwe cwe-022 \
    --input_file data/eval_tasks/seccodeplt_cwe022.jsonl \
    --output_dir results/double_steering/eval/cwe-022 \
    --safety_causal_results data/causal_analysis/cwe-022/head_causal_results.json \
    --safety_sv_dir data/steering_vectors/<model_slug>/<dataset_type>/cwe-022/response_mean/head/mean_diff \
    --correct_causal_results results/double_steering/causal_analysis/cwe-022/head_causal_results.json \
    --correct_sv_dir_orth results/double_steering/steering_vectors/cwe-022/orthogonal/head \
    --correct_sv_dir_raw  results/double_steering/steering_vectors/cwe-022/raw/head/mean_diff \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

Methods: `A` = safety + orthogonalized correctness at their own causal head sets; `E` = safety + raw correctness; `B` = joint head pool; `C` = correctness-only; `D` = single joint vector (requires `--joint_sv_dir`). Sweep grids come from `--alpha1_list/--alpha2_list` (per-direction strengths) and `--safety_topk_list/--correct_topk_list` (head budgets), falling back to `--alpha_list/--topk_list`. At heads selected by both sets the two offsets are simply added. Use `--skip_baseline/--skip_safety_only` to omit the reference conditions.

All commands take the model via `--model`. For another model such as Qwen-2.5-Coder-7B-Instruct, use method `E` (raw correctness vector, no orthogonalization), for example with `--alpha_list 1 3 5 --topk_list 16 64`, and point the vector and causal-ranking flags at that model's artifacts from Stages 2 and 3.
