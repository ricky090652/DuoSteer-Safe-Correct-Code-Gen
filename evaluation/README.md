# Stage 4 — Evaluation

Three instruments: CodeQL vulnerability rate `V`, GPT-4.1 functional-correctness rate `C` (joint score `C(1-V)`), and SecCodePLT execution-based unit tests.

Set `CODEQL_BIN` / `CODEQL_QLPACK` (or pass `--codeql` / `--qlpack_base`) and `OPENAI_API_KEY` first; see the top-level README.

## Vulnerability rate (CodeQL)

For plain generation files (any model output with `id`, `cwe_id`, `predicted_code`):

```bash
python evaluation/prepare_codeql_seccodeplt.py --input data/res_generations.jsonl --out_root data/codeql/my_run
python evaluation/run_codeql_seccodeplt.py --src_root data/codeql/my_run --out_root results/codeql_my_run
```

For DuoSteer output directories (per-condition JSONL files from `steering/duosteer_eval.py`):

```bash
python evaluation/codeql_eval_steered.py --cwe cwe-022 --eval_base results/double_steering/eval
```

For steering outputs organized by mode (`{steer_base}/{mode}/{cwe}/*.jsonl`, an alternative layout; this variant also drops syntactically invalid generations before counting):

```bash
python evaluation/codeql_eval_steered_modes.py --steer_base results/steering --cwe cwe-022 --mode safety_only double_E
```

Both wrap each snippet with the CWE-specific taint entry point before analysis and count a generation as vulnerable when the target-CWE query fires outside the injected wrapper (all CWEs, including CWE-079 whose v4 wrapper is source-only).

## Functional correctness (GPT-4.1 judge, Batch API)

The judge prompt is the CodeJudge-style template in `common/prompts.py` (`CODE_CORRECTNESS_EVALUATION`); it is instructed to ignore security issues so `V` and `C` measure disjoint failure modes.

```bash
# plain generation files
python evaluation/eval_correctness_seccodeplt.py prepare --input data/res_generations.jsonl --out_dir results/correctness/my_run
python evaluation/eval_correctness_seccodeplt.py submit  --out_dir results/correctness/my_run
python evaluation/eval_correctness_seccodeplt.py collect --input data/res_generations.jsonl --out_dir results/correctness/my_run

# steered condition grids
python evaluation/eval_correctness_steered.py prepare --steer_base results/double_steering/eval --out_dir results/correctness/steered
python evaluation/eval_correctness_steered.py submit  --out_dir results/correctness/steered
python evaluation/eval_correctness_steered.py collect --steer_base results/double_steering/eval --out_dir results/correctness/steered
```

## Execution-based unit tests (SecCodePLT)

Runs each generation against the SecCodePLT test suite in a hardened subprocess (`unit_test_runner.py`: no network, no shell, CPU/memory limits, throwaway working directory). Only the CWEs that ship test cases are covered.

```bash
# one-time sandbox check
python evaluation/unit_test_eval.py --selftest

python evaluation/unit_test_eval.py \
    --parquet /path/to/SecCodePLT/insecure_coding-00000-of-00001.parquet \
    --gen llama:079:baseline=results/double_steering/eval/cwe-079/baseline.jsonl \
    --gen llama:079:duosteer=results/double_steering/eval/cwe-079/<condition>.jsonl \
    --out_dir results/unit_test_eval
```

Each `--gen` spec is `MODEL:CWE:ARM=PATH` (repeatable). Download the parquet from the `Virtue-AI-HUB/SecCodePLT` dataset on Hugging Face. Run on a compute node, not a login node.
