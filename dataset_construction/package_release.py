"""
package_release.py

Stage 1, step 9: package a model's per-CWE pair files into the released
combined files and restore the full CodeQL finding records.

  output   <out_dir>/<file_prefix>_{intra,cross}.jsonl   (e.g. llama31-8b_intra.jsonl)
  id       codesec-<model_tag>-<tag>-<cwe>-<NNNN>         (e.g. codesec-llama31-intra-022-0001)
  fields   id, cwe_id, question, source, prompt, safe_code, vuln_code,
           vuln_codeql_detections  [+ structural_distance, fix_mechanism,
           annotation_rationale on INTRA only]
  finding  {queryName, startLine, startColumn, message, cweIds}
           startLine/startColumn are int or null (never ""), message is the
           CodeQL rule's fullDescription, cweIds come from the rule's cwe tags.

The pair files store compact detections ({query, cwe, line}). The full records
are recovered for every pair, in this order:
  1. this model's local CodeQL run (--local_codeql_dir: sarif/, chunks/, manifest.jsonl),
     in-code findings only (startLine <= the snippet's line count);
  2. a donor model's run (--donor_local_codeql_dir), for cross-model top-ups;
  3. a previous pair set (--prev_pairs_glob) whose records already carry full
     detections, for pairs kept through build_contrastive_pairs.py --prev_dir.
Findings are filtered to the pair's CWE. The build FAILS if any pair ends up with
no finding, or if any intra pair lacks annotations (unless --without_annotations
or --allow_missing_annotations).

Usage:
  python dataset_construction/package_release.py \
      --pairs_dir data/contrastive_pairs/llama --local_codeql_dir data/local_codeql/llama \
      --model_tag llama31 --file_prefix llama31-8b --out_dir data/contrastive_pairs
  python dataset_construction/package_release.py \
      --pairs_dir data/contrastive_pairs/qwen --local_codeql_dir data/local_codeql/qwen \
      --donor_local_codeql_dir data/local_codeql/llama \
      --model_tag qwen25 --file_prefix qwen25-coder-7b --out_dir data/contrastive_pairs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

CWES = ["022", "079", "094", "295", "502"]
ANN = ("structural_distance", "fix_mechanism", "annotation_rationale")
FENCE = re.compile(r"```(?:python)?\s*\n?(.*?)\n?```", re.S)


def clean(t: str) -> str:
    m = FENCE.search(t)
    return (m.group(1).strip() if m else t.strip())


def to_int_or_none(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# finding sources
# --------------------------------------------------------------------------- #

def load_sarif_dets(local_dir: Path) -> dict:
    """uid -> [full findings], in-code only (startLine <= manifest code_lines)."""
    code_lines = {}
    for line in open(local_dir / "manifest.jsonl"):
        m = json.loads(line)
        code_lines[m["uid"]] = m["code_lines"]
    dets = defaultdict(list)
    seen = set()
    for sf in sorted(glob.glob(str(local_dir / "sarif" / "*.sarif"))):
        data = json.load(open(sf))
        for run in data.get("runs", []):
            rules = run.get("tool", {}).get("driver", {}).get("rules", [])
            by_id = {r.get("id"): r for r in rules}
            for res in run.get("results", []):
                rule = by_id.get(res.get("ruleId")) or (
                    rules[res["ruleIndex"]] if "ruleIndex" in res and res["ruleIndex"] < len(rules) else {})
                qname = rule.get("properties", {}).get("id", res.get("ruleId", ""))
                msg = rule.get("fullDescription", {}).get("text") or rule.get("shortDescription", {}).get("text", "")
                cwe_ids = sorted(t.split("/")[-1] for t in rule.get("properties", {}).get("tags", [])
                                 if t.startswith("external/cwe/"))
                for loc in res.get("locations", []):
                    phys = loc.get("physicalLocation", {})
                    uid = os.path.splitext(os.path.basename(phys.get("artifactLocation", {}).get("uri", "")))[0]
                    reg = phys.get("region", {})
                    sl = to_int_or_none(reg.get("startLine"))
                    sc = to_int_or_none(reg.get("startColumn"))
                    if uid not in code_lines:
                        continue
                    if sl is not None and sl > code_lines[uid]:
                        continue                      # wrapper region: not the code's own sink
                    key = (uid, qname, sl, sc)
                    if key in seen:
                        continue
                    seen.add(key)
                    dets[uid].append({"queryName": qname, "startLine": sl, "startColumn": sc,
                                      "message": msg, "cweIds": cwe_ids})
    return dets


def load_chunk_code(local_dir: Path) -> dict:
    """clean code -> uid (from the run's chunk files)."""
    out = {}
    safe = re.compile(r"[^A-Za-z0-9_.-]+")
    for cf in glob.glob(str(local_dir / "chunks" / "*.jsonl")):
        for line in open(cf):
            r = json.loads(line)
            for i, c in enumerate(r["predicted_code"]):
                out.setdefault(c.strip(), safe.sub("-", f"{r['group']}__{r['id']}__g{i}"))
    return out


def load_prev_dets(prev_glob: str) -> dict:
    """clean vuln code -> normalized full findings, unioned across every previous
    record sharing that code (the same snippet can appear under several CWEs)."""
    out = defaultdict(list)
    seen = set()
    for fp in glob.glob(prev_glob):
        for line in open(fp):
            r = json.loads(line)
            c = clean(r["vuln_code"])
            for d in r.get("vuln_codeql_detections") or []:
                cwe_ids = d.get("cweIds")
                if not cwe_ids and d.get("cwe"):
                    cwe_ids = [f"cwe-{str(d['cwe']).zfill(3)}"]
                rec = {"queryName": d.get("queryName") or d.get("query", ""),
                       "startLine": to_int_or_none(d.get("startLine", d.get("line"))),
                       "startColumn": to_int_or_none(d.get("startColumn")),
                       "message": d.get("message", ""),
                       "cweIds": sorted(cwe_ids or [])}
                key = (c, rec["queryName"], rec["startLine"], rec["startColumn"])
                if key in seen:
                    continue
                seen.add(key)
                out[c].append(rec)
    return dict(out)


def pick(dets: list, cwe: str) -> list:
    tag = f"cwe-{cwe}"
    return [d for d in dets if tag in (d.get("cweIds") or [])]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def build(pairs_dir: Path, runs: list, prev_glob, model_tag: str, file_prefix: str,
          out_dir: Path, with_ann=True, allow_missing_ann=False):
    """runs: [(sarif_dets, code->uid), ...] own run first, donors after."""
    prev = load_prev_dets(prev_glob) if prev_glob else {}
    out_dir.mkdir(parents=True, exist_ok=True)

    report, problems = [], []
    src_stats = defaultdict(int)
    for tag in ("intra", "cross"):
        out_path = out_dir / f"{file_prefix}_{tag}.jsonl"
        total = 0
        with open(out_path, "w") as out:
            for cwe in CWES:
                src = pairs_dir / f"codesec_pairs_cwe-{cwe}_{tag}.jsonl"
                if not src.exists():
                    problems.append(f"{tag}/{cwe}: missing {src}")
                    continue
                rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
                for i, r in enumerate(rows, start=1):
                    c = clean(r["vuln_code"])
                    dets = None
                    for k, (sarif, chunk) in enumerate(runs):
                        if c in chunk:
                            dets = pick(sarif.get(chunk[c], []), cwe)
                            if dets:
                                src_stats["own_run" if k == 0 else "donor_run"] += 1
                                break
                    if not dets and c in prev:
                        dets = pick(prev[c], cwe); src_stats["previous_pairs"] += 1
                    if not dets:
                        problems.append(f"{tag}/{cwe} #{i}: no finding recovered")
                        dets = []
                    rec = {
                        "id": f"codesec-{model_tag}-{tag}-{cwe}-{i:04d}",
                        "cwe_id": cwe,
                        "question": r.get("question", ""),
                        "source": r.get("source", ""),
                        "prompt": r.get("prompt", ""),
                        "safe_code": r["safe_code"],
                        "vuln_code": r["vuln_code"],
                        "vuln_codeql_detections": dets,
                    }
                    if tag == "intra" and with_ann:
                        missing = [k for k in ANN if not r.get(k)]
                        if missing:
                            problems.append(f"intra/{cwe} #{i}: missing {missing}")
                        for k in ANN:
                            rec[k] = r.get(k)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += len(rows)
        report.append((out_path, total))

    print("=== files ===")
    for p, n in report:
        print(f"  {p}: {n}")
    print("=== finding sources ===")
    for s, n in sorted(src_stats.items()):
        print(f"  {s:15} {n}")
    n_ann = sum(1 for p in problems if "missing [" in p)
    n_det = sum(1 for p in problems if "no finding" in p)
    n_file = sum(1 for p in problems if "missing " in p and "missing [" not in p)
    print(f"=== problems: missing-file={n_file} missing-annotation={n_ann} no-finding={n_det} ===")
    for p in problems[:8]:
        print("   ", p)
    if n_file or n_det:
        sys.exit("FATAL: missing pair files or pairs without a recovered finding")
    if with_ann and n_ann and not allow_missing_ann:
        sys.exit("FATAL: intra pairs missing annotations (run annotate_pairs.py first), "
                 "or pass --without_annotations / --allow_missing_annotations")
    return [p for p, _ in report]


def validate(files):
    """Load every file with pyarrow and check the finding columns are typed int-or-null."""
    try:
        import pyarrow.json as pj
    except ImportError:
        print("pyarrow not installed; skipping type validation")
        return True
    ok = True
    for fp in files:
        tbl = pj.read_json(str(fp))
        rows = tbl.to_pylist()
        n_empty = sum(1 for r in rows if not r["vuln_codeql_detections"])
        bad = sum(1 for r in rows for d in r["vuln_codeql_detections"]
                  if not (d["startColumn"] is None or isinstance(d["startColumn"], int))
                  or not (d["startLine"] is None or isinstance(d["startLine"], int)))
        print(f"  {fp.name}: rows={tbl.num_rows} empty-finding rows={n_empty} bad-int columns={bad}")
        if n_empty or bad:
            ok = False
    print("VALIDATION", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pairs_dir", required=True, help="per-CWE pair files of one model")
    ap.add_argument("--local_codeql_dir", required=True, help="this model's local CodeQL run dir")
    ap.add_argument("--donor_local_codeql_dir", nargs="*", default=[],
                    help="other models' run dirs, for cross-model top-ups")
    ap.add_argument("--prev_pairs_glob", default=None,
                    help="previous pair files carrying full detections (for --prev_dir pairs)")
    ap.add_argument("--model_tag", required=True, help="id tag, e.g. llama31 or qwen25")
    ap.add_argument("--file_prefix", required=True, help="output file prefix, e.g. llama31-8b")
    ap.add_argument("--out_dir", default="data/contrastive_pairs")
    ap.add_argument("--without_annotations", action="store_true",
                    help="omit the three intra annotation fields entirely")
    ap.add_argument("--allow_missing_annotations", action="store_true",
                    help="write null annotation fields where missing (dry run)")
    ap.add_argument("--upload_repo", default=None,
                    help="optional Hugging Face dataset repo to upload the output files to")
    ap.add_argument("--commit_message", default="Update CodeSec-Pairs")
    a = ap.parse_args()

    runs = []
    for d in [a.local_codeql_dir] + list(a.donor_local_codeql_dir):
        d = Path(d)
        runs.append((load_sarif_dets(d), load_chunk_code(d)))
    files = build(Path(a.pairs_dir), runs, a.prev_pairs_glob, a.model_tag, a.file_prefix,
                  Path(a.out_dir), with_ann=not a.without_annotations,
                  allow_missing_ann=a.allow_missing_annotations)
    ok = validate(files)
    if a.upload_repo:
        if not ok or a.allow_missing_annotations:
            sys.exit("refusing to upload: validation failed or annotations missing")
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(Path(a.out_dir)), repo_id=a.upload_repo,
                              repo_type="dataset", commit_message=a.commit_message,
                              allow_patterns=[f"{a.file_prefix}_*.jsonl"])
        print(f"uploaded to {a.upload_repo}")


if __name__ == "__main__":
    main()
