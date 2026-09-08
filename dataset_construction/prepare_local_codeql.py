"""
prepare_local_codeql.py

Stage 1, step 5: package an extracted generation pool into a small,
self-contained folder that runs CodeQL chunk by chunk. No single database is
built over the whole pool (memory-safe) and the package stays small (split JSONL
plus scripts, not a half-million pre-wrapped files).

At run time `run_codeql.sh` processes one chunk at a time: `wrap_chunk.py` expands
that chunk's snippets into a temp dir (four wrapper trees), CodeQL builds one
database per tree, writes SARIF, and the temp dir and database are deleted before
the next chunk. Peak files on disk equal one chunk, not the whole pool.

Wrapper trees (see common/codeql_entry_points.py, wrapper v4):
  args   request.args -> every parameter        CWE-022 PathInjection + TarSlip, CWE-094 CodeInjection
  xss    source-only: request.args -> params,   CWE-079 ReflectedXss
         the wrapper returns a constant, so an XSS finding needs the code's own html sink
  deser  request.data -> first parameter        CWE-502 UnsafeDeserialization
  raw    no wrapper                             CWE-295 MissingHostKeyValidation + RequestWithoutValidation,
                                                CWE-079 Jinja2WithoutEscaping (structural)

Layout produced (per model):
  <out>/<model>/
    README.md
    codeql_entry_points.py     # copy of common/codeql_entry_points.py (imported by wrap_chunk.py)
    wrap_chunk.py              # one chunk JSONL -> temp <tree>/<uid>.py + manifest rows
    run_codeql.sh              # loop chunks x trees: wrap, db create, analyze, clean
    parse_codeql_results.py    # sarif/*.sarif -> labels.jsonl + labels_none.txt (in-code findings only)
    chunks/chunk_XXX.jsonl     # the extracted pool, split (~chunk_snippets per file)

Usage:
  python dataset_construction/prepare_local_codeql.py \
      --in_glob 'data/code_gen_extracted/llama/*.jsonl' \
      --model llama --out data/local_codeql --chunk_snippets 3000
"""
from __future__ import annotations

# --- repo path setup: allow running this script from any directory ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _d in ("common", "dataset_construction"):
    _p = _REPO_ROOT / _d
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# ---------------------------------------------------------------------
import argparse
import glob
import json
import os
import shutil
import stat

from extract_code import group_from_filename

WRAPPER_SRC = _REPO_ROOT / "common" / "codeql_entry_points.py"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--in_glob", required=True, help="extracted generation files (output of extract_code.py)")
    ap.add_argument("--model", required=True, help="short model name for the package dir (e.g. llama, qwen)")
    ap.add_argument("--out", default=str(_REPO_ROOT / "data" / "local_codeql"))
    ap.add_argument("--group", default=None, choices=["safe", "vuln", "vuln_generic"],
                    help="prompt group of ALL input files (default: inferred from each file name)")
    ap.add_argument("--chunk_snippets", type=int, default=3000,
                    help="target snippets per chunk (bounds each CodeQL database)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.in_glob))
    if not paths:
        raise SystemExit(f"no files match {args.in_glob}")
    root = os.path.join(args.out, args.model)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "chunks"))

    chunk_idx = 0
    cur_snips = 0
    n_records = 0
    n_snips = 0

    def open_chunk(i):
        return open(os.path.join(root, "chunks", f"chunk_{i:03d}.jsonl"), "w")

    cur = open_chunk(chunk_idx)
    for path in paths:
        g = args.group or group_from_filename(path)
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            preds = r.get("predicted_code", []) or []
            if isinstance(preds, str):
                preds = [preds]
            if not preds:
                continue
            rec = {"id": r["id"], "source": r.get("source", "?"),
                   "cwe_prompt": str(r.get("cwe_id", "0")), "group": g,
                   "predicted_code": preds}
            cur.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_records += 1
            n_snips += len(preds)
            cur_snips += len(preds)
            if cur_snips >= args.chunk_snippets:
                cur.close()
                chunk_idx += 1
                cur = open_chunk(chunk_idx)
                cur_snips = 0
    cur.close()
    n_chunks = chunk_idx + 1

    shutil.copy(WRAPPER_SRC, os.path.join(root, "codeql_entry_points.py"))
    _write(root, "wrap_chunk.py", WRAP_CHUNK)
    _write(root, "run_codeql.sh", RUN_SCRIPT, executable=True)
    _write(root, "parse_codeql_results.py", PARSER)
    _write(root, "README.md", readme_text(args.model, n_records, n_snips, n_chunks, args.chunk_snippets))

    print(f"[{args.model}] {n_records} records, {n_snips} snippets -> {n_chunks} chunk files "
          f"(~{args.chunk_snippets} snippets/chunk). Package: {root}")


def _write(root, name, content, executable=False):
    p = os.path.join(root, name)
    with open(p, "w") as fh:
        fh.write(content)
    if executable:
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


WRAP_CHUNK = '''"""Expand ONE chunk JSONL into temp wrapped files (4 trees) + manifest rows.
Usage: python wrap_chunk.py <chunk.jsonl> <workdir>
Writes <workdir>/<tree>/<uid>.py and <workdir>/manifest.jsonl.
"""
import sys, json, os, re
from codeql_entry_points import add_entry_point

# tree -> cwe used to pick the wrapper (None = unwrapped raw code)
TREES = {"args": "cwe-022", "xss": "cwe-079", "deser": "cwe-502", "raw": None}
SAFE = re.compile(r"[^A-Za-z0-9_.-]+")

chunk, workdir = sys.argv[1], sys.argv[2]
os.makedirs(workdir, exist_ok=True)
man = open(os.path.join(workdir, "manifest.jsonl"), "w")
for line in open(chunk):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    g, rid = r["group"], r["id"]
    for idx, code in enumerate(r["predicted_code"]):
        uid = SAFE.sub("-", f"{g}__{rid}__g{idx}")
        for tree, wc in TREES.items():
            d = os.path.join(workdir, tree)
            os.makedirs(d, exist_ok=True)
            wrapped = code if wc is None else add_entry_point(code, wc)
            with open(os.path.join(d, uid + ".py"), "w") as fh:
                fh.write(wrapped)
        man.write(json.dumps({"uid": uid, "record_id": rid, "source": r["source"],
                              "group": g, "gen_idx": idx,
                              "code_lines": len(code.splitlines())}) + "\\n")
man.close()
'''


RUN_SCRIPT = r'''#!/usr/bin/env bash
# Run CodeQL chunk-by-chunk. Each chunk is expanded to a temp dir, analyzed per
# wrapper tree, then deleted, so peak files/disk stays at one chunk.
#
# Required environment (same variables as the top-level README):
#   CODEQL_BIN     path to the codeql binary
#   CODEQL_QLPACK  path to .../qlpacks/codeql/python-queries/<version>/Security
# Optional: PYTHON (default python3), THREADS (default 4)
CODEQL="${CODEQL_BIN:-codeql}"
QLPACK="${CODEQL_QLPACK:?set CODEQL_QLPACK to .../qlpacks/codeql/python-queries/<version>/Security}"
PYTHON="${PYTHON:-python3}"
THREADS="${THREADS:-4}"

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p sarif _work

declare -A QUERIES=(
  ["args"]="CWE-022/PathInjection.ql CWE-022/TarSlip.ql CWE-094/CodeInjection.ql"
  ["xss"]="CWE-079/ReflectedXss.ql"
  ["deser"]="CWE-502/UnsafeDeserialization.ql"
  ["raw"]="CWE-295/MissingHostKeyValidation.ql CWE-295/RequestWithoutValidation.ql CWE-079/Jinja2WithoutEscaping.ql"
)

: > manifest.jsonl   # rebuilt on every run from the chunks (cheap, deterministic)
for chunk in chunks/chunk_*.jsonl; do
  cid=$(basename "$chunk" .jsonl)
  work="_work/$cid"
  rm -rf "$work"; mkdir -p "$work"
  "$PYTHON" wrap_chunk.py "$chunk" "$work"
  cat "$work/manifest.jsonl" >> manifest.jsonl

  # resume: skip the analysis when every tree's SARIF for this chunk exists
  done_all=1
  for tree in "${!QUERIES[@]}"; do [ -s "sarif/${cid}_${tree}.sarif" ] || done_all=0; done
  if [ "$done_all" -eq 1 ]; then echo "skip $cid (all sarif exist)"; rm -rf "$work"; continue; fi

  for tree in "${!QUERIES[@]}"; do
    out="sarif/${cid}_${tree}.sarif"
    [ -s "$out" ] && continue
    [ -d "$work/$tree" ] || continue
    qlist=""; for q in ${QUERIES[$tree]}; do qlist="$qlist $QLPACK/$q"; done
    echo "=== $cid/$tree : db + analyze ==="
    "$CODEQL" database create _db_tmp --language=python --source-root="$work/$tree" --overwrite --threads="$THREADS" >/dev/null
    "$CODEQL" database analyze _db_tmp $qlist --format=sarifv2.1.0 --output="$out" --no-print-metrics-summary --threads="$THREADS" >/dev/null
    rm -rf _db_tmp
  done
  rm -rf "$work"
done
rm -rf _work
echo "All chunks done. Now: $PYTHON parse_codeql_results.py"
'''


PARSER = r'''"""Parse sarif/*.sarif into labels.jsonl using manifest.jsonl.

A detection counts only when its startLine is INSIDE the generated code
(startLine <= code_lines); findings in the appended wrapper region are dropped.
Output: labels.jsonl (snippets with >= 1 in-code detection, with the studied CWE
of each finding) and labels_none.txt (snippets no studied query flagged).
"""
import glob, json, os
from collections import defaultdict

QUERY_CWE = {
    "py/path-injection": "022", "py/tarslip": "022",
    "py/code-injection": "094", "py/reflective-xss": "079",
    "py/jinja2/autoescape-false": "079",
    "py/unsafe-deserialization": "502",
    "py/paramiko-missing-host-key-validation": "295",
    "py/request-without-cert-validation": "295",
}
here = os.path.dirname(os.path.abspath(__file__))
code_lines, meta = {}, {}
for line in open(os.path.join(here, "manifest.jsonl")):
    m = json.loads(line); code_lines[m["uid"]] = m["code_lines"]; meta[m["uid"]] = m

def short_id(run, rule):
    for r in run.get("tool", {}).get("driver", {}).get("rules", []):
        if r.get("id") == rule:
            return r.get("properties", {}).get("id", rule)
    return rule

det = defaultdict(list)
for sf in sorted(glob.glob(os.path.join(here, "sarif", "*.sarif"))):
    data = json.load(open(sf))
    for run in data.get("runs", []):
        for res in run.get("results", []):
            rule = res.get("ruleId", "")
            q = short_id(run, rule)
            q = q if q in QUERY_CWE else rule
            for loc in res.get("locations", []):
                phys = loc.get("physicalLocation", {})
                uid = os.path.splitext(os.path.basename(
                    phys.get("artifactLocation", {}).get("uri", "")))[0]
                sl = phys.get("region", {}).get("startLine")
                if uid not in code_lines:
                    continue
                if sl is not None and sl > code_lines[uid]:
                    continue
                det[uid].append({"query": q, "cwe": QUERY_CWE.get(q, "?"), "line": sl})

with open(os.path.join(here, "labels.jsonl"), "w") as out:
    for uid, ds in det.items():
        seen, uniq = set(), []
        for d in ds:
            k = (d["query"], d["line"])
            if k in seen: continue
            seen.add(k); uniq.append(d)
        m = meta[uid]
        rec = {k: m[k] for k in ("uid", "record_id", "source", "group", "gen_idx")}
        rec["detections"] = uniq
        rec["cwes"] = sorted({d["cwe"] for d in uniq})
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open(os.path.join(here, "labels_none.txt"), "w") as out:
    for uid in code_lines:
        if uid not in det:
            out.write(uid + "\n")
print(f"labels.jsonl: {len(det)} flagged; {len(code_lines)-len(det)} clean (labels_none.txt)")
'''


def readme_text(model, n_records, n_snips, n_chunks, chunk_snippets):
    return f'''# Local CodeQL run: {model}

Self-contained, memory-safe CodeQL job for the extracted **{model}** generation
pool. The pool is split into {n_chunks} chunk files ({n_records:,} records / {n_snips:,}
snippets, ~{chunk_snippets:,} snippets per chunk). `run_codeql.sh` expands one chunk at
a time into a temp dir, runs CodeQL, then deletes it, so the whole pool is never
materialized on disk at once.

## Contents
- `chunks/chunk_XXX.jsonl`: the extracted generations (record = `{{id, source, cwe_prompt, group, predicted_code: [...]}}`)
- `wrap_chunk.py`: expands one chunk into `<tree>/<uid>.py` plus manifest rows
- `run_codeql.sh`: loops over chunks x trees; builds a DB, analyzes, cleans up
- `parse_codeql_results.py`: SARIF -> `labels.jsonl` + `labels_none.txt` (in-code findings only)
- `codeql_entry_points.py`: the v4 entry-point wrapper (copied from `common/`)

## Wrapper trees and queries
| tree | wrapper | queries |
|------|---------|---------|
| args | `args_string` (request.args -> every param) | CWE-022 PathInjection + TarSlip, CWE-094 CodeInjection |
| xss  | source-only (request.args -> params, returns a constant) | CWE-079 ReflectedXss |
| deser| `bytes_first` (request.data -> first param) | CWE-502 UnsafeDeserialization |
| raw  | none (unwrapped) | CWE-295 MissingHostKeyValidation + RequestWithoutValidation; CWE-079 Jinja2WithoutEscaping (structural, autoescape=False) |

CWE-079 is source-only: a detection requires the code's own html sink, so every
label is attributable to the generated code, like the other CWEs.
`parse_codeql_results.py` additionally drops any finding whose line falls in the
appended wrapper region.

## Setup
- CodeQL CLI with the `codeql/python-queries` pack (tested with CLI 2.25.2 and pack 1.8.0).
- Point the script at your install:
  ```bash
  export CODEQL_BIN=/path/to/codeql
  export CODEQL_QLPACK=/path/to/qlpacks/codeql/python-queries/<version>/Security
  export PYTHON=python3          # stdlib only; any Python 3.9+ works
  ```

## Run
```bash
bash run_codeql.sh              # resumable: skips chunks whose SARIF already exist
python parse_codeql_results.py
```
Each chunk builds four small databases (one per tree) and deletes each right after
its SARIF is written. Safe to interrupt and re-run.

## Outputs consumed by the pair builder
- `manifest.jsonl`: one row per snippet (`uid`, `record_id`, `source`, `group`, `gen_idx`, `code_lines`)
- `labels.jsonl`: one row per flagged snippet (`uid`, `record_id`, `source`, `group`, `gen_idx`, `detections`, `cwes`)
- `labels_none.txt`: clean snippets (safe-side candidates)
- `sarif/`: keep it; `package_release.py` reads the full finding records from it

Next: `python dataset_construction/build_contrastive_pairs.py --labels_dir <this dir> ...`
'''


if __name__ == "__main__":
    main()
