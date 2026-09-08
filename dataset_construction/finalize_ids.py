"""
finalize_ids.py

Normalize every pair id in a model's per-CWE pair set to one scheme:
    codesec-<model>-<cwe>-<kind>-<NNNN>
so ids are uniform regardless of origin (new run, re-validated previous pairs,
cross-model top-up). Run after building and any top-ups, before annotation.
(package_release.py assigns the final release ids, codesec-<tag>-<kind>-<cwe>-<NNNN>.)

Usage:
  python dataset_construction/finalize_ids.py --dir data/contrastive_pairs/qwen --model qwen
"""
import argparse
import json
import os

STUDIED = ["022", "079", "094", "295", "502"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dir", required=True, help="pair dir with codesec_pairs_cwe-XXX_{intra,cross}.jsonl")
    ap.add_argument("--model", required=True, help="model name to embed in the ids")
    a = ap.parse_args()
    total = 0
    for X in STUDIED:
        for kind in ("intra", "cross"):
            fp = os.path.join(a.dir, f"codesec_pairs_cwe-{X}_{kind}.jsonl")
            if not os.path.exists(fp):
                continue
            rows = [json.loads(l) for l in open(fp)]
            for i, r in enumerate(rows, 1):
                r["id"] = f"codesec-{a.model}-{X}-{kind}-{i:04d}"
            with open(fp, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(rows)
    print(f"{a.model}: re-id'd {total} pairs to codesec-{a.model}-<cwe>-<kind>-<NNNN>")


if __name__ == "__main__":
    main()
