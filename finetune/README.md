# DPO + LoRA baseline

A fine-tuning baseline to compare against CausalMD / DuoSteer. It is **not** the paper's
SFT baseline: the paper does full-parameter SFT on 5,000 safe∧correct samples (App. E),
only on Llama. Here we train one LoRA adapter with DPO on the same pairs DuoSteer uses.

| | Paper SFT | This baseline |
|---|---|---|
| Objective | SFT on safe∧correct code | DPO (sigmoid), reference = base model |
| Parameters | full | LoRA r=8, α=16, all attention + MLP projections |
| Data | 1,000 safe∧correct / CWE from CodeSec-Pairs | intra safety pairs + correctness pairs (the DuoSteer data), pooled over 5 CWEs |
| Prompt | p^b | p^b, via the chat template as in `steer_eval.py` |

## 1. Install (inside the container, once)

Keep the installed transformers version so the steering results stay comparable:

```bash
pip install peft trl datasets "transformers==$(python -c 'import transformers; print(transformers.__version__)')"
```

## 2. Build the preference data

```bash
python finetune/build_dpo_data.py --pair_prefix qwen25-coder-7b \
    --correctness_dir data/double_steering/qwen_merged/correctness_pairs \
    --out_dir data/dpo/qwen
```

Split is question-level (90/10, seed 42); SecCodePLT questions are dropped.
`--include_cross` also adds the cross-prompt pairs (not used by DuoSteer).
For Qwen: 2,626 train / 291 val pairs (520 / 59 questions).

## 3. Train

```bash
python finetune/train_dpo_lora.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --data_dir data/dpo/qwen --output_dir results/qwen/dpo_lora/run1
```

Defaults: β=0.1, lr 5e-5, effective batch 16, 3 epochs, cosine, warmup 0.1,
max_length 2048 / prompt 1024. One checkpoint and one val evaluation per epoch;
`log_history.json` holds `eval_rewards/accuracies` and `eval_rewards/margins`.
Add `--gradient_checkpointing` if memory is short.

## 4. Evaluate

Generate with the adapter merged into the base model, then score with the usual
CodeQL / GPT-4.1 / `summarize_steering.py` steps under mode `dpo_lora`:

```bash
python steering/steer_eval.py --setting baseline --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter results/qwen/dpo_lora/run1 \
    --input_file data/eval_tasks/seccodeplt_cwe022.jsonl \
    --output_dir results/qwen/steering/dpo_lora/cwe-022
```
