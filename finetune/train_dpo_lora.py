"""
DPO + LoRA fine-tuning baseline (not in the DuoSteer paper; the paper's SFT baseline is
full-parameter SFT, see paper App. E).

Trains one LoRA adapter on the pooled preference data from build_dpo_data.py. The
reference model is the base model (TRL disables the adapter for the reference pass).
Prompts are rendered with the tokenizer's chat template exactly as steer_eval.py does
(single user turn, no explicit system message), so training and evaluation see the same
prompt string.

Outputs in --output_dir: checkpoint-*/ (one per epoch), the final adapter, tokenizer,
run_config.json and log_history.json (train/eval loss, rewards/accuracies, rewards/margins).

Ported from prosec-prefix/train_prefix.py (LoRA + DPO path only).

Usage:
  python finetune/train_dpo_lora.py \\
      --model Qwen/Qwen2.5-Coder-7B-Instruct \\
      --data_dir data/dpo/qwen --output_dir results/qwen/dpo_lora/run1
"""
import argparse
import dataclasses
import inspect
import json
import os

import torch
# Some torch builds do not auto-import torch.distributed.tensor, which newer peft reads
# when wrapping nn.Linear with LoRA.
try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

LORA_TARGETS = {
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def load_split(path: str, tokenizer):
    ds = load_dataset("json", data_files=path, split="train")

    def fmt(ex):
        ex["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        return ex

    keep = {"prompt", "chosen", "rejected"}
    return ds.map(fmt, remove_columns=[c for c in ds.column_names if c not in keep])


def make_config(args) -> DPOConfig:
    """Build DPOConfig, dropping fields this TRL/transformers version does not have."""
    want = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        beta=args.beta,
        loss_type="sigmoid",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=True,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        seed=args.seed,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        evaluation_strategy="epoch",   # older transformers name
        report_to=[],
        remove_unused_columns=False,
    )
    fields = {f.name for f in dataclasses.fields(DPOConfig)}
    dropped = sorted(k for k in want if k not in fields)
    if dropped:
        print(f"DPOConfig: ignoring unsupported fields {dropped}")
    return DPOConfig(**{k: v for k, v in want.items() if k in fields})


class DumpLogHistory(TrainerCallback):
    """Write log_history.json at every save and at the end."""

    def _dump(self, args, state):
        with open(os.path.join(args.output_dir, "log_history.json"), "w") as f:
            json.dump(state.log_history, f, indent=1)

    def on_save(self, args, state, control, **kw):
        self._dump(args, state)

    def on_train_end(self, args, state, control, **kw):
        self._dump(args, state)


def print_eval_summary(log_history: list):
    rows = [h for h in log_history if "eval_loss" in h]
    if not rows:
        return
    print("\nepoch  step  eval_loss  eval_acc  eval_margin")
    for h in rows:
        print(f"{h.get('epoch', 0):5.2f} {h.get('step', 0):5d}  {h['eval_loss']:9.4f}  "
              f"{h.get('eval_rewards/accuracies', float('nan')):8.3f}  "
              f"{h.get('eval_rewards/margins', float('nan')):11.4f}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "run_config.json"), "w"), indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    targets = (LORA_TARGETS.get(model.config.model_type, "all-linear")
               if args.lora_target_modules == "auto"
               else [m.strip() for m in args.lora_target_modules.split(",") if m.strip()])
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=targets, bias="none",
    )

    train_ds = load_split(os.path.join(args.data_dir, "train.jsonl"), tokenizer)
    val_ds = load_split(os.path.join(args.data_dir, "val.jsonl"), tokenizer)
    steps_per_epoch = -(-len(train_ds) // (args.batch_size * args.grad_accum))
    print(f"train {len(train_ds)} pairs, val {len(val_ds)} pairs, "
          f"~{steps_per_epoch} steps/epoch x {args.epochs} epochs")
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} targets={targets}; "
          f"DPO beta={args.beta} lr={args.lr} eff_batch={args.batch_size * args.grad_accum}")

    kwargs = dict(model=model, ref_model=None, args=make_config(args),
                  train_dataset=train_ds, eval_dataset=val_ds, peft_config=peft_cfg)
    sig = inspect.signature(DPOTrainer.__init__).parameters
    kwargs["processing_class" if "processing_class" in sig else "tokenizer"] = tokenizer
    trainer = DPOTrainer(**kwargs)
    trainer.add_callback(DumpLogHistory())
    trainer.model.print_trainable_parameters()

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print_eval_summary(trainer.state.log_history)
    print(f"\nDone. Final adapter in {args.output_dir}; per-epoch adapters in checkpoint-*/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--data_dir", default="data/dpo/qwen", help="Dir with train.jsonl / val.jsonl")
    ap.add_argument("--output_dir", required=True)
    # LoRA (ProSec App. C: r=8, alpha=16)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_target_modules", default="auto",
                    help="Comma-separated module names, or 'auto' (all attention + MLP projections)")
    # DPO / optimisation
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=0.3)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
