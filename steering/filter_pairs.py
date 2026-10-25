"""
Write the ids of contrastive pairs whose safe and vuln code use the same libraries.

Pairs where only one side imports a library mostly contrast "which library" rather than
"how to use it safely" (e.g. paramiko+AutoAddPolicy vs. a requests-based rewrite), which
dilutes the mean-difference direction. The output feeds
`extract_steering_vector.py --keep_ids`.

Example:
  python steering/filter_pairs.py --pairs_file data/codesec_pairs/qwen25-coder-7b_intra.jsonl \\
      --cwe 295 --libs paramiko requests --out data/pair_filters/qwen25-coder-7b_intra_cwe-295_same_lib.txt
"""
import argparse
import json
import re
from pathlib import Path


def uses(code: str, lib: str) -> bool:
    return re.search(rf"(^|\n)\s*(import|from)\s+{re.escape(lib)}\b", code) is not None


def main(args):
    pairs = [json.loads(l) for l in open(args.pairs_file) if l.strip()]
    cwe = args.cwe.replace("cwe-", "")
    pairs = [p for p in pairs if p["cwe_id"] == cwe]
    keep = [p for p in pairs
            if all(uses(p["safe_code"], lib) == uses(p["vuln_code"], lib) for lib in args.libs)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(p["id"] + "\n" for p in keep))
    n_q = len({p["question"] for p in keep})
    print(f"CWE-{cwe}: kept {len(keep)} / {len(pairs)} pairs ({n_q} questions) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pairs_file", required=True)
    ap.add_argument("--cwe", required=True, help="e.g. 295 or cwe-295")
    ap.add_argument("--libs", nargs="+", required=True, help="libraries that must appear on both sides or neither")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
