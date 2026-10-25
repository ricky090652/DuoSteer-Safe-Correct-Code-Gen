"""
CWE-022 augmentation pairs for pathlib-style tasks.

The intra-prompt pipeline cannot pair these tasks: under the benign prompt the
model almost never writes a containment check (EXPERIMENTS §5.24). Here the safe
side is sampled under CODE_GENERATION_PROMPT_SAFE_PATH (benign template plus a
requirement naming the os.path.realpath + startswith check) and the vulnerable
side under the benign prompt. Representations are later extracted with the
benign prompt for both sides, so the prompt text itself does not enter them.

  prepare  write the two generation inputs (each question repeated N times)
  build    after generation + CodeQL, pair CodeQL-clean safe samples with
           CodeQL-flagged benign samples and append them to the CWE's intra pairs

Example:
  python dataset_construction/augment_cwe022.py prepare --questions data/augment/cwe022_pathlib_questions.jsonl
  python dataset_construction/augment_cwe022.py build \\
      --vuln_gen results/qwen/augment/aug_vuln/cwe-022/baseline.jsonl \\
      --safe_gen results/qwen/augment/aug_safe/cwe-022/baseline.jsonl \\
      --codeql_issues results/qwen/eval_augment/codeql/cwe-022/issues_target.json \\
      --base_pairs data/codesec_pairs/qwen25-coder-7b_intra.jsonl \\
      --out data/codesec_pairs/qwen25-coder-7b_intra_aug022.jsonl
"""
# --- repo path setup: allow running this script from any directory ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _d in ("common",):
    _p = _REPO_ROOT / _d
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# ---------------------------------------------------------------------

import argparse
import ast
import io
import json
import re
import tokenize
from collections import defaultdict
from pathlib import Path

from prompts import CODE_GENERATION_PROMPT, CODE_GENERATION_PROMPT_SAFE_PATH

FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)\n?```", re.DOTALL)
CHECK_RE = re.compile(r"os\.path\.(realpath|normpath)")


def extract_code(text: str) -> str:
    m = FENCE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def strip_comments(code: str) -> str:
    """Drop # comments (and lines left empty by them); keep everything else verbatim."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    lines = code.splitlines()
    for t in reversed([t for t in toks if t.type == tokenize.COMMENT]):
        row, col = t.start
        lines[row - 1] = lines[row - 1][:col].rstrip()
    return "\n".join(l for i, l in enumerate(lines)
                     if l.strip() or not code.splitlines()[i].strip()).strip()


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def cmd_prepare(args):
    qs = [json.loads(l) for l in open(args.questions) if l.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.questions).stem
    with open(out / f"{stem}_x{args.n}_vuln.jsonl", "w") as fv, open(out / f"{stem}_x{args.n}_safe.jsonl", "w") as fs:
        for q in qs:
            for j in range(args.n):
                rid = f"{q['id']}__gen{j}"
                fv.write(json.dumps({**q, "id": rid}) + "\n")
                msg = CODE_GENERATION_PROMPT_SAFE_PATH.format(question=q["question"])
                fs.write(json.dumps({**q, "id": rid, "messages": [{"role": "user", "content": msg}]}) + "\n")
    print(f"{len(qs)} questions x {args.n} -> {out}/{stem}_x{args.n}_{{vuln,safe}}.jsonl")


def flagged_map(issues_file: str) -> dict:
    """(mode, sample id) -> list of CodeQL detections."""
    out = defaultdict(list)
    for r in json.load(open(issues_file)).get("Results", []):
        cond_dir, fname = r["fileName"].split("/")[-2:]
        mode = cond_dir.split("__")[0]
        out[(mode, fname[:-3])].extend(r.get("CWEs", []))
    return out


def cmd_build(args):
    flags = flagged_map(args.codeql_issues)
    cand = defaultdict(lambda: {"safe": {}, "vuln": {}})
    stats = defaultdict(lambda: defaultdict(int))
    for side, path, mode in (("vuln", args.vuln_gen, args.vuln_mode), ("safe", args.safe_gen, args.safe_mode)):
        for line in open(path):
            r = json.loads(line)
            qid = r["id"].split("__")[0]
            code = strip_comments(extract_code(r["predicted_code"]))
            if not parses(code):
                stats[qid][f"{side}_syntax_error"] += 1
                continue
            det = flags.get((mode, re.sub(r"[^\w\-]", "_", r["id"])), [])
            if side == "vuln" and det:
                cand[qid]["vuln"].setdefault(code, (r, det))
            elif side == "safe" and not det and CHECK_RE.search(code) and "startswith" in code:
                cand[qid]["safe"].setdefault(code, (r, det))
            else:
                stats[qid][f"{side}_rejected"] += 1
    base = [json.loads(l) for l in open(args.base_pairs) if l.strip()]
    base = [p for p in base if p["cwe_id"] == args.cwe]
    new, k = [], 0
    for qid in sorted(cand):
        safes, vulns = list(cand[qid]["safe"].items()), list(cand[qid]["vuln"].items())
        n = min(args.max_per_question, max(len(safes), len(vulns))) if safes and vulns else 0
        for i in range(n):
            (s_code, (s_rec, _)), (v_code, (v_rec, v_det)) = safes[i % len(safes)], vulns[i % len(vulns)]
            k += 1
            new.append({
                "id": f"codesec-qwen25-aug{args.cwe}-{k:04d}", "cwe_id": args.cwe,
                "question": s_rec["question"], "source": "augment-pathlib",
                "prompt": CODE_GENERATION_PROMPT.format(question=s_rec["question"]),
                "safe_prompt": "CODE_GENERATION_PROMPT_SAFE_PATH",
                "safe_code": f"```python\n{s_code}\n```", "vuln_code": f"```python\n{v_code}\n```",
                "vuln_codeql_detections": v_det,
            })
        print(f"{qid}: safe {len(safes)} / vuln {len(vulns)} distinct -> {n} pairs  {dict(stats[qid])}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in base + new:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_q = len({p["question"] for p in new})
    print(f"CWE-{args.cwe}: {len(base)} original + {len(new)} new pairs ({n_q} questions) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--questions", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out_dir", default="data/augment")
    b = sub.add_parser("build")
    b.add_argument("--vuln_gen", required=True)
    b.add_argument("--safe_gen", required=True)
    b.add_argument("--vuln_mode", default="aug_vuln")
    b.add_argument("--safe_mode", default="aug_safe")
    b.add_argument("--codeql_issues", required=True)
    b.add_argument("--base_pairs", required=True)
    b.add_argument("--cwe", default="022")
    b.add_argument("--max_per_question", type=int, default=5)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_prepare(a) if a.cmd == "prepare" else cmd_build(a)
