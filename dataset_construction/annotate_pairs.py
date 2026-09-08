"""
annotate_pairs.py

Stage 1, step 8: annotate the intra-prompt pairs with structural_distance and
fix_mechanism using GPT-4.1 through the OpenAI Batch API, then embed the labels
into the pair files (intra files directly; a cross pair sharing the same
safe_code/vuln_code inherits the label).

Reads <pair_dir>/codesec_pairs_cwe-XXX_intra.jsonl (the per-CWE files written by
build_contrastive_pairs.py and finalize_ids.py).

Outputs under --out_dir:
  batch_input.jsonl    submitted requests
  batch_state.json     batch id and status, for --resume
  batch_output.jsonl   raw API output
  categories.jsonl     one record per pair with the parsed labels

Usage:
  export OPENAI_API_KEY=...
  python dataset_construction/annotate_pairs.py --pair_dir data/contrastive_pairs/llama --submit
  # or step by step: --prepare, then --resume (poll + parse + embed) or --parse
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
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from prompts import build_annotation_messages

CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]
MODEL = "gpt-4.1"
ANN = ("structural_distance", "fix_mechanism", "annotation_rationale")


def build_request(pair: dict) -> dict:
    """One Batch API request; the annotation prompt is rendered from common/prompts.py."""
    return {
        "custom_id": pair["custom_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": build_annotation_messages(pair["cwe"], pair["vuln_code"], pair["safe_code"]),
            "temperature": 0,
            "max_tokens": 200,
        },
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_intra_pairs(pdir: Path) -> list[dict]:
    pairs = []
    for cwe in CWES:
        f = pdir / f"codesec_pairs_{cwe}_intra.jsonl"
        if not f.exists():
            print(f"  [warn] missing {f}")
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            qn = [d.get("query") or d.get("queryName") or ""
                  for d in rec.get("vuln_codeql_detections", [])]
            dominant = max(set(qn), key=qn.count) if qn else "unknown"
            pairs.append({
                "custom_id": rec["id"],
                "cwe": cwe,
                "pair_id": rec["id"],
                "query_name": dominant,
                "vuln_code": rec["vuln_code"],
                "safe_code": rec["safe_code"],
            })
    return pairs


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def prepare(pairs, out: Path, batch_input: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    with open(batch_input, "w") as f:
        for p in pairs:
            f.write(json.dumps(build_request(p)) + "\n")
    print(f"Wrote {len(pairs)} requests -> {batch_input} ({batch_input.stat().st_size/1024:.0f} KB)")
    return len(pairs)


def get_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set")
    from openai import OpenAI
    return OpenAI(api_key=key)


def submit(client, batch_input: Path, state_file: Path) -> str:
    with open(batch_input, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  Uploaded file_id={uploaded.id}")
    batch = client.batches.create(input_file_id=uploaded.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window="24h")
    print(f"  Batch submitted: {batch.id}  status={batch.status}")
    state_file.write_text(json.dumps({"batch_id": batch.id, "status": batch.status}, indent=2))
    return batch.id


def poll(client, batch_id: str, state_file: Path) -> str:
    interval = 30
    while True:
        batch = client.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"  [{batch.status}] completed={c.completed} failed={c.failed} total={c.total}", flush=True)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            state_file.write_text(json.dumps({"batch_id": batch_id, "status": batch.status}, indent=2))
            if batch.status != "completed":
                sys.exit(f"Batch ended with status={batch.status}")
            return batch.output_file_id
        time.sleep(interval)
        interval = min(interval * 1.5, 300)


def download(client, output_file_id: str, batch_output: Path):
    content = client.files.content(output_file_id)
    batch_output.write_bytes(content.content)
    print(f"  Saved -> {batch_output} ({batch_output.stat().st_size/1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Parse + embed
# ---------------------------------------------------------------------------

def parse_and_embed(pairs, pdir: Path, batch_output: Path, categories_out: Path):
    results = {}
    with open(batch_output) as f:
        for line in f:
            row = json.loads(line)
            cid = row["custom_id"]
            if row.get("error"):
                results[cid] = {"parse_error": True, "error": str(row["error"])}
                continue
            choices = row.get("response", {}).get("body", {}).get("choices", [])
            if not choices:
                results[cid] = {"parse_error": True, "error": "no choices"}
                continue
            raw = choices[0]["message"]["content"].strip()
            try:
                results[cid] = json.loads(raw)
            except json.JSONDecodeError:
                results[cid] = {"parse_error": True, "raw": raw}

    n_ok = n_err = 0
    with open(categories_out, "w") as f:
        for p in pairs:
            r = results.get(p["custom_id"], {"parse_error": True, "error": "missing"})
            f.write(json.dumps({
                "custom_id": p["custom_id"], "cwe": p["cwe"], "pair_id": p["pair_id"],
                "query_name": p["query_name"],
                "structural_distance": r.get("structural_distance"),
                "fix_mechanism": r.get("fix_mechanism"),
                "rationale": r.get("rationale"),
                "parse_error": r.get("parse_error", False),
            }) + "\n")
            n_err += bool(r.get("parse_error")); n_ok += not bool(r.get("parse_error"))
    print(f"Parsed {n_ok} OK, {n_err} errors -> {categories_out}")

    labels, by_code = {}, {}
    for p in pairs:
        r = results.get(p["custom_id"], {})
        if not r.get("parse_error") and r.get("structural_distance"):
            labels[p["pair_id"]] = (r["structural_distance"], r["fix_mechanism"], r.get("rationale"))
            by_code[(p["safe_code"], p["vuln_code"])] = p["pair_id"]
    for cwe in CWES:
        for kind, bc in (("intra", None), ("cross", by_code)):
            fp = pdir / f"codesec_pairs_{cwe}_{kind}.jsonl"
            if fp.exists():
                embed_file(fp, labels, bc)

    print("\n=== category distribution (intra) ===")
    recs = [json.loads(l) for l in open(categories_out)]
    for cwe in CWES:
        rl = [r for r in recs if r["cwe"] == cwe and not r["parse_error"]]
        sd = Counter(r["structural_distance"] for r in rl)
        fm = Counter(r["fix_mechanism"] for r in rl)
        print(f"  {cwe}: n={len(rl)}  dist={dict(sd)}  fix={dict(fm)}")


def embed_file(path: Path, labels: dict, by_code):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = 0
    for rec in rows:
        lab = labels.get(rec["id"])
        if lab is None and by_code is not None:
            pid2 = by_code.get((rec["safe_code"], rec["vuln_code"]))
            lab = labels.get(pid2) if pid2 else None
        if lab:
            rec["structural_distance"], rec["fix_mechanism"], rec["annotation_rationale"] = lab
            n += 1
    with open(path, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  embedded {n}/{len(rows)} -> {path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true", help="build batch_input.jsonl only")
    g.add_argument("--submit", action="store_true", help="prepare + submit + poll + parse + embed")
    g.add_argument("--resume", action="store_true", help="resume polling from batch_state.json")
    g.add_argument("--parse", action="store_true", help="parse an existing batch_output.jsonl and embed")
    ap.add_argument("--pair_dir", required=True,
                    help="dir with codesec_pairs_cwe-XXX_{intra,cross}.jsonl")
    ap.add_argument("--out_dir", default=None,
                    help="batch files and categories.jsonl (default: results/category_analysis/<pair_dir name>)")
    a = ap.parse_args()

    pdir = Path(a.pair_dir)
    out = Path(a.out_dir) if a.out_dir else Path("results/category_analysis") / pdir.name
    state_file, batch_input = out / "batch_state.json", out / "batch_input.jsonl"
    batch_output, categories_out = out / "batch_output.jsonl", out / "categories.jsonl"

    pairs = load_intra_pairs(pdir)
    print(f"Loaded {len(pairs)} intra pairs from {pdir}")
    for cwe in CWES:
        print(f"  {cwe}: {sum(1 for p in pairs if p['cwe'] == cwe)}")

    if a.prepare:
        prepare(pairs, out, batch_input)
    elif a.submit:
        prepare(pairs, out, batch_input)
        client = get_client()
        bid = submit(client, batch_input, state_file)
        ofid = poll(client, bid, state_file)
        download(client, ofid, batch_output)
        parse_and_embed(pairs, pdir, batch_output, categories_out)
    elif a.resume:
        if not state_file.exists():
            sys.exit(f"{state_file} not found")
        bid = json.loads(state_file.read_text())["batch_id"]
        print(f"Resuming batch {bid}")
        client = get_client()
        ofid = poll(client, bid, state_file)
        download(client, ofid, batch_output)
        parse_and_embed(pairs, pdir, batch_output, categories_out)
    elif a.parse:
        if not batch_output.exists():
            sys.exit(f"{batch_output} not found")
        parse_and_embed(pairs, pdir, batch_output, categories_out)


if __name__ == "__main__":
    main()
