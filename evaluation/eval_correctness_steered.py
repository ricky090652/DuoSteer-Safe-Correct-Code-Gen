"""
GPT-4.1 correctness judge for Qwen/Llama steered outputs across an (alpha, top_k) grid.

Reads:  results/{run_root}/steering/{mode}/{cwe}/*.jsonl
        (one file per (alpha, top_k) condition; records have predicted_code + question)

Writes per-cwe labeled outputs and a summary JSON of correctness rates per
(mode, condition).

Subcommands (same as eval_correctness_seccodeplt.py):
  prepare  --steer_base RESULTS_ROOT  --out_dir BATCH_DIR
  submit   --out_dir BATCH_DIR
  collect  --steer_base RESULTS_ROOT  --out_dir BATCH_DIR

Example:
  python eval_correctness_steered.py prepare \
      --steer_base results/steering --out_dir results/correctness/steered
  python eval_correctness_steered.py submit \
      --out_dir results/correctness/steered
  python eval_correctness_steered.py collect \
      --steer_base results/steering --out_dir results/correctness/steered
"""
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
import re
from collections import defaultdict
from pathlib import Path

from prompts import CODE_CORRECTNESS_EVALUATION

CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]
MODEL = "gpt-4.1"
SEVERITY_ORDER = {"Negligible": 0, "Small": 1, "Major": 2, "Fatal": 3}


def extract_code(text):
    if isinstance(text, list):
        text = text[0] if text else ""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text or "", re.DOTALL)
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def parse_gpt_response(content):
    judgments = []
    try:
        text = content.strip()
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        raw_json = m.group(1) if m else text
        start = raw_json.find("[")
        if start >= 0:
            parsed, _ = json.JSONDecoder().raw_decode(raw_json, start)
        else:
            parsed = json.loads(raw_json)
        if isinstance(parsed, list) and parsed:
            judgments = [
                {"inconsistency": str(j.get("inconsistency", "None")),
                 "severity": str(j.get("severity", "Negligible"))}
                for j in parsed if isinstance(j, dict)
            ]
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass
    if not judgments:
        for block in re.split(r"\n\s*\n", content):
            inc_m = re.search(r"Inconsistency:\s*(.+)", block, re.I)
            sev_m = re.search(r"Severity:\s*(.+)", block, re.I)
            if inc_m:
                inc = inc_m.group(1).strip()
                sev = sev_m.group(1).strip() if sev_m else "Negligible"
                judgments.append({"inconsistency": inc, "severity": sev})
    if not judgments:
        return True, "Negligible", content
    correct = True
    for j in judgments:
        sev = j.get("severity", "").strip()
        if sev in ("Fatal", "Major") or sev not in ("", "None", "Negligible", "Small"):
            correct = False
            break
    severity = max((j.get("severity", "Negligible") for j in judgments),
                   key=lambda s: SEVERITY_ORDER.get(s, 0))
    return correct, severity, content


def iter_steer_files(steer_base, modes):
    """Yield (mode, cwe, condition, path) for every condition JSONL."""
    base = Path(steer_base)
    for mode_dir in (m for m in modes if (base / m).is_dir()):
        for cwe in CWES:
            cwe_dir = base / mode_dir / cwe
            if not cwe_dir.is_dir():
                continue
            for f in sorted(cwe_dir.glob("*.jsonl")):
                yield mode_dir, cwe, f.stem, f


def cmd_prepare(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch_input_gpt41.jsonl"

    n = 0
    per_cwe = defaultdict(int)
    with open(out_path, "w") as fout:
        for mode, cwe, cond, fpath in iter_steer_files(args.steer_base, args.modes):
            with open(fpath) as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    question = r.get("question", "")
                    code = extract_code(r.get("predicted_code", ""))
                    custom_id = f"{mode}__{cwe}__{cond}__{r['id']}"
                    prompt = CODE_CORRECTNESS_EVALUATION \
                        .replace("{PROBLEM}", question) \
                        .replace("{CODE}", code)
                    fout.write(json.dumps({
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0,
                            "max_tokens": 1024,
                        }
                    }) + "\n")
                    n += 1
                    per_cwe[cwe] += 1
    print(f"prepared {n} batch records → {out_path}")
    for c in CWES:
        print(f"  {c}: {per_cwe[c]}")


def cmd_submit(args):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out_dir = Path(args.out_dir)
    batch_input = out_dir / "batch_input_gpt41.jsonl"
    if not batch_input.exists():
        raise SystemExit(f"missing {batch_input} — run prepare first")
    print(f"uploading {batch_input}...")
    with open(batch_input, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"creating batch (file_id={upload.id})...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    state = {"batch_id": batch.id, "input_file_id": upload.id,
             "status": batch.status}
    with open(out_dir / "batch_state.json", "w") as f:
        json.dump(state, f, indent=2)
    print(f"batch_id={batch.id}  status={batch.status}")


def cmd_collect(args):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out_dir = Path(args.out_dir)
    state_path = out_dir / "batch_state.json"
    if not state_path.exists():
        raise SystemExit("no batch_state.json — submit first")
    state = json.load(open(state_path))
    batch = client.batches.retrieve(state["batch_id"])
    print(f"batch_id={batch.id}  status={batch.status}  "
          f"completed={batch.request_counts.completed}/"
          f"{batch.request_counts.total}")
    state["status"] = batch.status
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    if batch.status != "completed":
        return
    if batch.output_file_id is None:
        print("no output file")
        return

    raw_bytes = client.files.content(batch.output_file_id).content
    lines = raw_bytes.decode().strip().split("\n")
    print(f"downloading {len(lines)} results...")

    # Build (mode, cwe, cond, id) -> orig record index
    idx = {}
    for mode, cwe, cond, fpath in iter_steer_files(args.steer_base, args.modes):
        with open(fpath) as fin:
            for line in fin:
                if not line.strip():
                    continue
                r = json.loads(line)
                idx[(mode, cwe, cond, r["id"])] = r

    labeled_path = out_dir / "labeled.jsonl"
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0})))
    n = 0
    with open(labeled_path, "w") as fout:
        for line in lines:
            if not line.strip():
                continue
            resp = json.loads(line)
            cid = resp.get("custom_id", "")
            parts = cid.split("__", 3)
            if len(parts) != 4:
                continue
            mode, cwe, cond, rid = parts
            body = resp.get("response", {}).get("body", {})
            content = ""
            if body.get("choices"):
                content = body["choices"][0].get("message", {}).get("content", "")
            correct, severity, raw = parse_gpt_response(content)
            orig = idx.get((mode, cwe, cond, rid), {})
            rec = {
                "id": rid, "cwe_id": cwe, "mode": mode, "condition": cond,
                "src_id": orig.get("src_id", ""),
                "predicted_code": orig.get("predicted_code", ""),
                "gpt41_correct": correct, "gpt41_severity": severity,
                "gpt41_raw": raw,
            }
            fout.write(json.dumps(rec) + "\n")
            summary[cwe][mode][cond]["total"] += 1
            if correct:
                summary[cwe][mode][cond]["correct"] += 1
            n += 1
    print(f"wrote {n} labeled records → {labeled_path}")

    rates = {}
    for cwe, by_mode in summary.items():
        rates[cwe] = {}
        for mode, by_cond in by_mode.items():
            rates[cwe][mode] = {}
            for cond, s in by_cond.items():
                rate = s["correct"] / s["total"] if s["total"] else 0.0
                rates[cwe][mode][cond] = {
                    "n_total": s["total"], "n_correct": s["correct"],
                    "correctness_rate": round(rate, 4),
                }
    rates_path = out_dir / "correctness_rates.json"
    with open(rates_path, "w") as f:
        json.dump(rates, f, indent=2)
    print(f"wrote {rates_path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "submit", "collect"):
        sp = sub.add_parser(name)
        sp.add_argument("--out_dir", required=True)
        if name != "submit":
            sp.add_argument("--steer_base", required=True)
            sp.add_argument("--modes", nargs="+", default=["safety_only", "double_A"])
    args = p.parse_args()
    {"prepare": cmd_prepare, "submit": cmd_submit, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
