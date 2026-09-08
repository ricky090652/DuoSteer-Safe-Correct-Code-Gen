"""
build_contrastive_pairs.py

Stage 1, step 6: extract safe-vs-vulnerable contrastive pairs from a completed
local CodeQL run (prepare_local_codeql.py package, after run_codeql.sh and
parse_codeql_results.py).

Pair definitions:
  intra  same benign prompt on both sides: vuln = a benign-prompt generation that
         CodeQL flagged for CWE X; safe = a benign-prompt generation of the same
         question that no studied query flagged.
  cross  vuln = a generation from a vulnerability-eliciting prompt (vuln or
         vuln_generic group) flagged for X; safe = a clean benign-prompt generation
         of the same question.

Rules enforced here:
  1. Completeness of BOTH sides. Every safe_code and vuln_code passes the same
     completeness gate as extraction (compiles, not a bare fragment, no
     undefined local names in def-free scripts). Candidates that fail are dropped.
  2. Targeted tasks anchor on their target CWE. A task that carries a target CWE
     (e.g. CyberSecEval-Instruct, SecurityEval, CodeLMSec) contributes pairs only
     for that CWE, and only when CodeQL actually flagged the vulnerable side for
     it, so prompt intent and CodeQL label agree. Targets outside the studied set
     contribute nothing.
  3. Untargeted tasks use their CodeQL label as the CWE. Tasks without a target
     CWE (Emergent-Misalignment, cwe_id 0) take whatever studied CWE CodeQL
     flagged. A question may then serve several CWEs; --max_cwes_per_question caps
     that overlap and prefers the emptier CWEs.
  4. The label maps to the exact snippet shown. The vulnerable side's detection is
     an in-code finding on THAT snippet for THAT CWE (wrapper-region findings were
     already dropped by parse_codeql_results.py). The safe side is in labels_none
     (flagged by no studied query). The matching detections are carried into
     vuln_codeql_detections.
  5. Dedup: no duplicate vuln_code per (question, CWE); the safe pool is cycled so
     safe_code is reused only when there are fewer clean generations than
     vulnerable ones.
  6. Optional: keep valid previous pairs (--prev_dir). An earlier pair set for the
     same model is re-validated against rules 1 and 4 and used to top up any
     (CWE, kind) bucket that is below its target. When a previous snippet is also
     in the new pool, the new run's label is authoritative; otherwise its stored
     detections must reference the CWE, and for CWE-079 the code must contain its
     own html sink.

Output: <out_dir>/codesec_pairs_cwe-<NNN>_{intra,cross}.jsonl with fields
  id, cwe_id ("022" style), question, source, prompt (rendered at run time from
  the templates in common/prompts.py), safe_code, vuln_code
  (both fenced ```python blocks), vuln_codeql_detections [{query, cwe, line}].
Ids are provisional; run finalize_ids.py next.

Usage:
  python dataset_construction/build_contrastive_pairs.py --model llama \
      --labels_dir data/local_codeql/llama \
      --code_glob 'data/code_gen_extracted/llama/*.jsonl' \
      --out_dir data/contrastive_pairs/llama
  # capped set, e.g. 300 intra / 200 cross per CWE
  python dataset_construction/build_contrastive_pairs.py --model qwen ... \
      --cap_intra 300 --cap_cross 200
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
import re
from collections import defaultdict

from extract_code import strip_fences, is_complete, group_from_filename
from prompts import build_generation_prompt, load_cwe_db

STUDIED = ["022", "079", "094", "295", "502"]
STUDIED_SET = set(STUDIED)
SAFE_UID = re.compile(r"[^A-Za-z0-9_.-]+")
HTML = re.compile(r'render_template_string|render_template|make_response|Markup|'
                  r'Response\(|<[a-zA-Z/][^>]*>|text/html', re.I)
GROUP_PROMPT_TYPE = {"safe": "code_gen", "vuln": "code_gen_vuln", "vuln_generic": "code_gen_vuln_generic"}


def norm_cwe(x) -> str:
    """'cwe-22' / '22' / 22 -> '022'; unspecified -> '000'."""
    s = re.sub(r"[^0-9]", "", str(x))
    s = s.lstrip("0")
    return s.zfill(3) if s else "000"


def uid_for(group, rid, gen_idx):
    return SAFE_UID.sub("-", f"{group}__{rid}__g{gen_idx}")


def fenced(code: str) -> str:
    return "```python\n" + code + "\n```"


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def load_labels(labels_dir):
    """uid -> set of studied CWEs flagged in-code; uid -> detections; set of clean uids."""
    flagged, detmap = {}, {}
    for line in open(os.path.join(labels_dir, "labels.jsonl")):
        r = json.loads(line)
        cwes = {norm_cwe(c) for c in r.get("cwes", [])} & STUDIED_SET
        if cwes:
            flagged[r["uid"]] = cwes
            detmap[r["uid"]] = r.get("detections", [])
    clean = set()
    np_ = os.path.join(labels_dir, "labels_none.txt")
    if os.path.exists(np_):
        clean = {l.strip() for l in open(np_) if l.strip()}
    return flagged, detmap, clean


def load_pool(code_glob, group_override=None):
    """uid -> {code, rid, source, cwe_prompt, group, gen_idx}; rid -> task record
    (question, cwe_id, vulnerability_type/description if present)."""
    pool, task = {}, {}
    paths = sorted(glob.glob(code_glob))
    if not paths:
        raise SystemExit(f"no files match {code_glob}")
    for path in paths:
        g = group_override or group_from_filename(path)
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r["id"]
            task.setdefault(rid, {k: r.get(k) for k in
                                  ("question", "cwe_id", "vulnerability_type", "vulnerability_description")})
            src = r.get("source", "?")
            cwe_prompt = norm_cwe(r.get("cwe_id", "0"))
            preds = r.get("predicted_code", []) or []
            if isinstance(preds, str):
                preds = [preds]
            for i, code in enumerate(preds):
                pool[uid_for(g, rid, i)] = {"code": code, "rid": rid, "source": src,
                                            "cwe_prompt": cwe_prompt, "group": g, "gen_idx": i}
    return pool, task



# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def complete_code(raw):
    """Clean code if it passes the completeness gate, else None."""
    c = strip_fences(raw)
    return c if is_complete(c) else None


def build(model, labels_dir, code_glob, prev_dir, out_dir,
          cap_intra=0, cap_cross=0, max_cwes_per_q=2, match_prev_counts=False,
          untargeted_sources=("emergent-misalignment",), group_override=None):
    flagged, detmap, clean = load_labels(labels_dir)
    pool, task = load_pool(code_glob, group_override)
    cwe_db = load_cwe_db()
    question = {rid: (tk.get("question") or "") for rid, tk in task.items()}
    untargeted = set(untargeted_sources)
    rid_meta = {}
    for m in pool.values():
        rid_meta.setdefault(m["rid"], (m["source"], m["cwe_prompt"]))

    # per-question candidate buckets (uids), completeness-gated (rule 1)
    safe_clean = defaultdict(list)                       # q -> [uid]
    safe_flag = defaultdict(lambda: defaultdict(list))   # q -> X -> [uid]
    cross_flag = defaultdict(lambda: defaultdict(list))  # q -> X -> [uid]
    code_cache = {}
    n_incomplete = 0
    for uid, m in pool.items():
        c = complete_code(m["code"])
        if c is None:
            n_incomplete += 1
            continue
        code_cache[uid] = c
        q = m["rid"]
        if m["group"] == "safe":
            if uid in clean:
                safe_clean[q].append(uid)
            elif uid in flagged:
                for X in flagged[uid]:
                    safe_flag[q][X].append(uid)
        else:
            if uid in flagged:
                for X in flagged[uid]:
                    cross_flag[q][X].append(uid)

    def q_cwes(q):
        src, cwe_prompt = rid_meta.get(q, (None, "000"))
        if src not in untargeted and cwe_prompt != "000":
            return [cwe_prompt] if cwe_prompt in STUDIED_SET else []   # rule 2
        found = set(safe_flag[q].keys()) | set(cross_flag[q].keys())    # rule 3
        return sorted(found & STUDIED_SET)

    def get_prompt(kind, q, grp):
        """Render the prompt of the vulnerable side's group from the task (run time)."""
        g = "safe" if kind == "intra" else grp
        tk = task.get(q, {})
        return build_generation_prompt(
            tk.get("question") or "", GROUP_PROMPT_TYPE[g], cwe_id=tk.get("cwe_id", "0"),
            cwe_db=cwe_db, vulnerability_type=tk.get("vulnerability_type"),
            vulnerability_description=tk.get("vulnerability_description"))

    pairs = {X: {"intra": [], "cross": []} for X in STUDIED}
    q_cwe_used = defaultdict(set)

    def emit(kind, X, q, vuln_uid, safe_uid):
        m = pool[vuln_uid]
        return {
            "id": f"codesec-{model}-{X}-{kind}-{q}-g{m['gen_idx']}",
            "cwe_id": X, "question": question.get(q, ""),
            "source": m["source"], "prompt": get_prompt(kind, q, m["group"]),
            "safe_code": fenced(code_cache[safe_uid]),
            "vuln_code": fenced(code_cache[vuln_uid]),
            "vuln_codeql_detections": [d for d in detmap.get(vuln_uid, [])
                                       if norm_cwe(d.get("cwe")) == X],
        }

    for q in sorted(safe_flag.keys() | cross_flag.keys()):
        cwes = q_cwes(q)
        if max_cwes_per_q and len(cwes) > max_cwes_per_q:   # overlap reduction (rule 3)
            cwes = sorted(cwes, key=lambda X: len(pairs[X]["intra"]) + len(pairs[X]["cross"]))[:max_cwes_per_q]
        for X in cwes:
            clean_uids = list(dict.fromkeys(safe_clean[q]))
            if not clean_uids:
                continue
            for kind, bucket in (("intra", safe_flag), ("cross", cross_flag)):
                seen_v, si = set(), 0
                for vuid in bucket[q].get(X, []):
                    vc = code_cache[vuid]
                    if vc in seen_v:                          # rule 5
                        continue
                    seen_v.add(vc)
                    suid = clean_uids[si % len(clean_uids)]; si += 1
                    pairs[X][kind].append(emit(kind, X, q, vuid, suid))
                    q_cwe_used[q].add(X)

    # new-run maps so previous pairs can be judged by the authoritative new labels
    newcode_cwes, newcode_dets, newcode_clean = defaultdict(set), {}, set()
    for uid, c in code_cache.items():
        if uid in flagged:
            newcode_cwes[c] |= flagged[uid]
            newcode_dets.setdefault(c, detmap.get(uid, []))
        elif uid in clean:
            newcode_clean.add(c)

    def prev_count(X, kind):
        fp = os.path.join(prev_dir or "", f"codesec_pairs_cwe-{X}_{kind}.jsonl")
        return sum(1 for _ in open(fp)) if os.path.exists(fp) else 0

    caps = {}
    for X in STUDIED:
        for kind in ("intra", "cross"):
            caps[(X, kind)] = prev_count(X, kind) if match_prev_counts \
                else (cap_intra if kind == "intra" else cap_cross)
    kept_prev = defaultdict(lambda: defaultdict(int))
    short = []
    for X in STUDIED:
        for kind in ("intra", "cross"):
            cap = caps[(X, kind)]
            lst = pairs[X][kind]
            if cap and len(lst) >= cap:
                pairs[X][kind] = lst[:cap]
            elif prev_dir and os.path.isdir(prev_dir):
                need = (cap - len(lst)) if cap else None
                kept_prev[X][kind] = _topup_previous(
                    prev_dir, X, kind, pairs[X][kind], need,
                    newcode_cwes, newcode_dets, newcode_clean)
            if cap and len(pairs[X][kind]) < cap:
                short.append((X, kind, len(pairs[X][kind]), cap))

    os.makedirs(out_dir, exist_ok=True)
    for X in STUDIED:
        for kind in ("intra", "cross"):
            fp = os.path.join(out_dir, f"codesec_pairs_cwe-{X}_{kind}.jsonl")
            with open(fp, "w") as f:
                for rec in pairs[X][kind]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n=== {model} contrastive pairs ===")
    print(f"snippets in pool: {len(pool)}  (dropped as incomplete: {n_incomplete})")
    print(f"{'cwe':6}{'intra':>8}{'cross':>8}{'kept_prev(i/c)':>18}")
    for X in STUDIED:
        kp = kept_prev[X]
        print(f"{X:6}{len(pairs[X]['intra']):>8}{len(pairs[X]['cross']):>8}"
              f"{str(kp.get('intra',0))+'/'+str(kp.get('cross',0)):>18}")
    multi = sum(1 for s in q_cwe_used.values() if len(s) > 1)
    print(f"questions used: {len(q_cwe_used)}; serving >1 CWE: {multi}")
    if short:
        print("BELOW TARGET (top up with topup_cross_model.py if wanted):")
        for X, kind, have_n, cap in short:
            print(f"  cwe-{X} {kind}: {have_n}/{cap}")
    print(f"written to {out_dir}")


def _norm_dets(dets, X):
    """Any detection schema -> [{query, cwe, line}], filtered to X."""
    out = []
    for d in dets:
        cwes = {norm_cwe(c) for c in ([d.get("cwe")] + (d.get("cweIds") or [])) if c}
        if X in cwes:
            out.append({"query": d.get("query") or d.get("queryName") or "",
                        "cwe": X, "line": d.get("line", d.get("startLine"))})
    return out


def _topup_previous(prev_dir, X, kind, existing, need, newcode_cwes, newcode_dets, newcode_clean):
    """Fill an under-target (X, kind) bucket from a previous pair set (rule 6).
    Returns how many pairs were added. need=None takes every valid pair."""
    fp = os.path.join(prev_dir, f"codesec_pairs_cwe-{X}_{kind}.jsonl")
    if not os.path.exists(fp):
        return 0
    have = {p["vuln_code"] for p in existing}
    added = 0
    for line in open(fp):
        if need is not None and added >= need:
            break
        r = json.loads(line)
        sc = complete_code(r.get("safe_code", "")); vc = complete_code(r.get("vuln_code", ""))
        if sc is None or vc is None:                              # rule 1
            continue
        if (vc in newcode_cwes) or (vc in newcode_clean):         # snippet in the new pool
            if X not in newcode_cwes.get(vc, set()):
                continue
            if sc not in newcode_clean:
                continue
            dets = _norm_dets(newcode_dets.get(vc, []), X)
        else:                                                     # absent from the new run
            dets = _norm_dets(r.get("vuln_codeql_detections") or [], X)
            if not dets:                                          # rule 4
                continue
            if X == "079" and not HTML.search(vc):                # own html sink required
                continue
        if fenced(vc) in have:
            continue
        rec = {
            "id": r.get("id", f"prev-{X}-{kind}-{added}"),
            "cwe_id": X, "question": r.get("question", ""),
            "source": r.get("source", "?"), "prompt": r.get("prompt", ""),
            "safe_code": fenced(sc), "vuln_code": fenced(vc),
            "vuln_codeql_detections": dets,
        }
        have.add(rec["vuln_code"])
        existing.append(rec)
        added += 1
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", required=True, help="short model name embedded in provisional ids")
    ap.add_argument("--labels_dir", required=True,
                    help="local CodeQL run dir (manifest.jsonl, labels.jsonl, labels_none.txt)")
    ap.add_argument("--code_glob", required=True, help="extracted generation files (output of extract_code.py)")
    ap.add_argument("--group", default=None, choices=["safe", "vuln", "vuln_generic"],
                    help="prompt group of ALL --code_glob files (default: inferred from file names)")
    ap.add_argument("--prev_dir", default=None, help="earlier pair set to re-validate and merge (rule 6)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cap_intra", type=int, default=0, help="target intra pairs per CWE (0 = keep all)")
    ap.add_argument("--cap_cross", type=int, default=0, help="target cross pairs per CWE (0 = keep all)")
    ap.add_argument("--max_cwes_per_question", type=int, default=2,
                    help="overlap cap for untargeted questions (0 = no cap)")
    ap.add_argument("--match_prev_counts", action="store_true",
                    help="per-CWE target = the count in --prev_dir")
    ap.add_argument("--untargeted_sources", default="emergent-misalignment",
                    help="comma-separated sources whose tasks carry no target CWE")
    a = ap.parse_args()
    build(a.model, a.labels_dir, a.code_glob, a.prev_dir, a.out_dir,
          a.cap_intra, a.cap_cross, a.max_cwes_per_question, a.match_prev_counts,
          tuple(s.strip() for s in a.untargeted_sources.split(",") if s.strip()), a.group)


if __name__ == "__main__":
    main()
