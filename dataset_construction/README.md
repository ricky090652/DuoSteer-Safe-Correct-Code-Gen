# Stage 1: CodeSec-Pairs Construction

Builds the contrastive safe-vs-vulnerable pair dataset from scratch: normalize source benchmarks, sample completions, extract complete code snippets, label them with CodeQL, pair safe and vulnerable generations, annotate the pairs, and package the release files.

Run all commands from the repository root (any working directory works; defaults resolve against the repo root). CodeQL runs in a separate, self-contained package (step 4) that can be executed on any machine with a CodeQL CLI, so the GPU host never needs CodeQL.

## 1. Normalize source benchmarks

```bash
python dataset_construction/dataset_preprocess.py \
    -i /path/to/securityeval_dataset.jsonl \
    -o data/preprocessed/securityeval.jsonl \
    -dname securityeval
```

`-dname` choices: `securityeval`, `codelmsec`, `seccodeplt`, `cyberseceval`, `emergent-misalignment`. Concatenate the normalized files into one task list per split as needed. The generation pool covers Emergent-Misalignment, CyberSecEval-Instruct, SecurityEval, and CodeLMSec; SecCodePLT is held out for evaluation.

## 2. Sample generations

Ten samples per task with sampling decoding. The input is a task file (`id`, `cwe_id`, `question`, `source`, `src_id`); the prompt is rendered from `question` at run time with the template selected by `--prompt_type`, so no prompt files are stored:

```bash
# Benign prompt p^b
python dataset_construction/generate_code.py \
    -i data/preprocessed/tasks.jsonl -o data/code_gen_results_sampling/res_benign.jsonl \
    -m meta-llama/Meta-Llama-3.1-8B-Instruct --prompt_type code_gen \
    --num_samples 10 --do_sample --temperature 1.0 --top_p 0.95

# Vulnerability-eliciting prompt p^e (CWE-specific from the task's cwe_id; generic when the task has no CWE)
python dataset_construction/generate_code.py \
    -i data/preprocessed/tasks.jsonl -o data/code_gen_results_sampling/res_vuln.jsonl \
    -m meta-llama/Meta-Llama-3.1-8B-Instruct --prompt_type code_gen_vuln \
    --num_samples 10 --do_sample --temperature 1.0 --top_p 0.95

# Generic vulnerability-eliciting prompt for every task
python dataset_construction/generate_code.py \
    -i data/preprocessed/tasks.jsonl -o data/code_gen_results_sampling/res_vuln_generic.jsonl \
    -m meta-llama/Meta-Llama-3.1-8B-Instruct --prompt_type code_gen_vuln_generic \
    --num_samples 10 --do_sample --temperature 1.0 --top_p 0.95
```

All prompt templates live in `common/prompts.py` (`build_generation_prompt`); the CWE-specific template takes the CWE name and description from `data/cwe_official/all_cwe.json`. Keep `benign` (or `safe`), `vuln`, and `vuln_generic` in the output file names: later steps read the prompt group from the file name (or take it from `--group`). The generator works with any Hugging Face chat model; pass a different `-m` (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct`) to collect pairs from another model. Requires one GPU; the script resumes if re-run with its own output as input.

## 3. Extract complete code snippets

```bash
python dataset_construction/extract_code.py \
    --in_glob 'data/code_gen_results_sampling/res_*.jsonl' \
    --out_dir data/code_gen_extracted/llama
```

Extracts the code from each generation (strips markdown fences) and keeps only **complete standalone** snippets. A snippet is complete when it compiles as a module (so a bare `return` outside a function fails), and, if it defines no function or class, references no undefined local name. For code-completion tasks (a `## COMPLETE CODE HERE` template), a bare completion is spliced back into the template at the right indentation and kept if the result is complete. Anything still incomplete is dropped. A missing import or an app-level global is not a reason to drop, since the CodeQL wrapper supplies taint through parameters. Add `--dry_run` for counts only, `--examples N` to print spliced samples. Output keeps the input schema, with a `predicted_code_meta` status per sample.

## 4. Label with CodeQL (self-contained, chunked package)

```bash
python dataset_construction/prepare_local_codeql.py \
    --in_glob 'data/code_gen_extracted/llama/*.jsonl' \
    --model llama --out data/local_codeql --chunk_snippets 3000
```

Writes `data/local_codeql/llama/`: the pool split into chunk files, a copy of `common/codeql_entry_points.py`, and three scripts. Copy or move that directory to any machine with a CodeQL CLI and run:

```bash
export CODEQL_BIN=/path/to/codeql
export CODEQL_QLPACK=/path/to/qlpacks/codeql/python-queries/<version>/Security
bash run_codeql.sh              # one chunk at a time; resumable
python parse_codeql_results.py  # sarif/*.sarif -> labels.jsonl + labels_none.txt
```

For each chunk, `wrap_chunk.py` writes four trees of wrapped files and `run_codeql.sh` builds one database per tree, so memory and disk stay bounded by one chunk:

| tree | wrapper (`common/codeql_entry_points.py`, v4) | queries |
|---|---|---|
| `args` | `request.args` fed into every parameter | CWE-022 `PathInjection`, `TarSlip`; CWE-094 `CodeInjection` |
| `xss` | source-only: `request.args` fed into the parameters, the wrapper returns a constant | CWE-079 `ReflectedXss` |
| `deser` | `request.data` fed into the first parameter | CWE-502 `UnsafeDeserialization` |
| `raw` | none | CWE-295 `MissingHostKeyValidation`, `RequestWithoutValidation`; CWE-079 `Jinja2WithoutEscaping` |

The wrapper only supplies a taint **source**; it never renders a return value. An XSS finding therefore needs an html sink inside the generated code, exactly as the other CWEs need their `open`, `eval`, or `pickle.loads` inside the code. Parameters are read from the AST, so annotated signatures (including `Dict[str, Callable[[str], str]]`) and `async def` wrap correctly. `parse_codeql_results.py` keeps a finding only when its line lies inside the generated code (the manifest records each snippet's line count), so nothing in the appended wrapper region can label a snippet. Bring back `manifest.jsonl`, `labels.jsonl`, `labels_none.txt`, and the `sarif/` directory. Tested with CodeQL CLI 2.25.2 and `codeql/python-queries` 1.8.0.

## 5. Build contrastive pairs

```bash
python dataset_construction/build_contrastive_pairs.py --model llama \
    --labels_dir data/local_codeql/llama \
    --code_glob 'data/code_gen_extracted/llama/*.jsonl' \
    --out_dir data/contrastive_pairs/llama
```

Writes `data/contrastive_pairs/llama/codesec_pairs_cwe-XXX_{intra,cross}.jsonl`. Intra-prompt pairs (safe and vulnerable samples from the same benign prompt) train probes and steering vectors; cross-prompt pairs (vulnerable side from a vulnerability-eliciting prompt) are used only for causal head knockout. The builder enforces:

- both sides pass the completeness gate of step 3;
- a task with a target CWE (CyberSecEval-Instruct, SecurityEval, CodeLMSec) contributes pairs only for that CWE and only when CodeQL flagged the vulnerable side for it; a task without one (Emergent-Misalignment) takes the CWE CodeQL flagged, with `--max_cwes_per_question` capping how many CWEs one question serves;
- the vulnerable side's finding is an in-code finding for that CWE on that snippet, carried into `vuln_codeql_detections`; the safe side was flagged by no studied query;
- no duplicate vulnerable code per (question, CWE); safe code is reused only when a question has fewer clean generations than vulnerable ones.

`--cap_intra N --cap_cross N` cap each CWE (e.g. 300 / 200 for a smaller replication set); `--prev_dir` re-validates an earlier pair set for the same model against the same rules and merges it. Then normalize the ids:

```bash
python dataset_construction/finalize_ids.py --dir data/contrastive_pairs/llama --model llama
```

If a capped set is short for some CWE, `topup_cross_model.py --into data/contrastive_pairs/qwen --donor data/contrastive_pairs/llama --cap_intra 300 --cap_cross 200` fills it from another model's pairs (code is model-agnostic; the added pairs are validated the same way). Run `finalize_ids.py` again afterwards.

## 6. Annotate pairs (structural distance and fix mechanism)

GPT-4.1 via the OpenAI Batch API, intra-prompt pairs only:

```bash
export OPENAI_API_KEY=...
python dataset_construction/annotate_pairs.py --pair_dir data/contrastive_pairs/llama --submit
# or step-by-step: --prepare, then --resume / --parse
```

The annotation prompt and few-shot examples are in `common/prompts.py` (`build_annotation_messages`). Labels are embedded into the intra files (`structural_distance`: `MINIMAL/REFACTOR/DIVERGENT`; `fix_mechanism`: `DELETION/SUBSTITUTION/ADDITION-GUARD/ADDITION-CONFIG/UNCLEAR`; `annotation_rationale`), and a cross pair that shares the same two snippets inherits them. Batch files and `categories.jsonl` go to `results/category_analysis/<model>/`.

## 7. Package the release files

```bash
python dataset_construction/package_release.py \
    --pairs_dir data/contrastive_pairs/llama --local_codeql_dir data/local_codeql/llama \
    --model_tag llama31 --file_prefix llama31-8b --out_dir data/contrastive_pairs
python dataset_construction/verify_codesec_pairs.py --pair_dir data/contrastive_pairs
```

Produces `data/contrastive_pairs/llama31-8b_{intra,cross}.jsonl` with the release ids (`codesec-llama31-intra-022-0001`) and restores the full CodeQL finding records (`queryName`, `startLine`, `startColumn`, `message`, `cweIds`) from the run's SARIF, in-code findings only. Pass `--donor_local_codeql_dir` for pairs added by `topup_cross_model.py`, and `--prev_pairs_glob` for pairs merged from an earlier set. The build fails if any pair has no recoverable finding or any intra pair lacks annotations.

## Evaluation-side CodeQL scripts

`run_codeql_detection.py` and `format_output_new.py` in this directory serve Stage 4: they run the same target-CWE queries over per-condition steering outputs and convert SARIF to JSON (`format_output_new.py -i issues.sarif -o issues.json --source_dir DIR` drops findings inside the injected wrapper). See `evaluation/README.md`.
