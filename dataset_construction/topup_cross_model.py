"""
topup_cross_model.py

Optional Stage 1 step: fill any under-target (CWE, kind) bucket of one model's
pair set from another model's pair set. Code is model-agnostic once generated, so
a donor pair is valid for the target model's dataset as long as it is complete,
its detections reference the CWE, and it is not a duplicate. Only the missing
count is taken, up to the per-kind cap. Donor pairs are re-id'd to the target
model; run finalize_ids.py afterwards to renumber everything consistently.

Usage:
  python dataset_construction/topup_cross_model.py \
      --into data/contrastive_pairs/qwen --donor data/contrastive_pairs/llama \
      --cap_intra 300 --cap_cross 200
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

from extract_code import strip_fences, is_complete

STUDIED = ["022", "079", "094", "295", "502"]


def valid(rec):
    for side in ("safe_code", "vuln_code"):
        if not is_complete(strip_fences(rec.get(side, ""))):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--into", required=True, help="pair dir to fill (codesec_pairs_cwe-XXX_{intra,cross}.jsonl)")
    ap.add_argument("--donor", required=True, help="pair dir to take pairs from")
    ap.add_argument("--model", default=None,
                    help="target model name used in re-id'd ids (default: basename of --into)")
    ap.add_argument("--cap_intra", type=int, default=300)
    ap.add_argument("--cap_cross", type=int, default=200)
    a = ap.parse_args()
    caps = {"intra": a.cap_intra, "cross": a.cap_cross}
    into_model = a.model or os.path.basename(os.path.normpath(a.into))

    for X in STUDIED:
        for kind in ("intra", "cross"):
            cap = caps[kind]
            into_fp = os.path.join(a.into, f"codesec_pairs_cwe-{X}_{kind}.jsonl")
            donor_fp = os.path.join(a.donor, f"codesec_pairs_cwe-{X}_{kind}.jsonl")
            cur = [json.loads(l) for l in open(into_fp)] if os.path.exists(into_fp) else []
            if len(cur) >= cap or not os.path.exists(donor_fp):
                continue
            need = cap - len(cur)
            have = {r["vuln_code"] for r in cur}
            have_ids = {r["id"] for r in cur}
            added = 0
            for line in open(donor_fp):
                if added >= need:
                    break
                r = json.loads(line)
                if r["vuln_code"] in have:
                    continue
                if not valid(r):
                    continue
                # the detection must reference X (the donor guarantees this; re-check)
                dc = {re.sub(r"[^0-9]", "", str(d.get("cwe", ""))).zfill(3)
                      for d in (r.get("vuln_codeql_detections") or [])}
                if X not in dc:
                    continue
                new_id = re.sub(r"^codesec-[^-]+-", f"codesec-{into_model}-", r["id"])
                if new_id in have_ids:
                    new_id = f"{new_id}-x{added}"
                r["id"] = new_id
                have.add(r["vuln_code"]); have_ids.add(new_id)
                cur.append(r)
                added += 1
            if added:
                with open(into_fp, "w") as f:
                    for r in cur:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"cwe-{X} {kind}: +{added} from donor -> {len(cur)}/{cap}")

    print("\n=== final counts in", a.into, "===")
    for X in STUDIED:
        row = []
        for kind in ("intra", "cross"):
            fp = os.path.join(a.into, f"codesec_pairs_cwe-{X}_{kind}.jsonl")
            n = sum(1 for _ in open(fp)) if os.path.exists(fp) else 0
            row.append(f"{kind}={n}")
        print(f"  cwe-{X}: " + ", ".join(row))


if __name__ == "__main__":
    main()
