"""
V / C / C(1-V) from several independent GPT-4.1 judge runs of the same outputs.

The judge is not deterministic: identical code gets a different strict verdict
in about 5-12% of repeated judgments, often enough to decide a 1-2 question
comparison. Here each generation is judged in every run, C counts a generation
as correct when the majority of runs say all severities are Negligible, and
the per-run C values are printed to show the spread.

Usage:
  python evaluation/summarize_rejudge.py \\
      --codeql_rates results/qwen/eval_rejudge/codeql/detection_rates.json \\
      --correctness_dirs results/qwen/eval_rejudge/correctness_r1 \\
                         results/qwen/eval_rejudge/correctness_r2 \\
                         results/qwen/eval_rejudge/correctness_r3
"""
import argparse
import json
from collections import defaultdict

from summarize_steering import SEVERITY_RE


def load_verdicts(labeled_path):
    """(cwe, mode, condition) -> {item id: strict verdict}."""
    out = defaultdict(dict)
    with open(labeled_path) as f:
        for line in f:
            r = json.loads(line)
            sevs = SEVERITY_RE.findall(r.get("gpt41_raw") or "")
            ok = bool(sevs) and all(s.strip() == "Negligible" for s in sevs)
            out[(r["cwe_id"], r["mode"], r["condition"])][r["id"]] = ok
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--codeql_rates", required=True)
    ap.add_argument("--correctness_dirs", nargs="+", required=True,
                    help="One collect output directory per judge run")
    args = ap.parse_args()

    with open(args.codeql_rates) as f:
        codeql_rates = json.load(f)
    runs = [load_verdicts(f"{d}/labeled.jsonl") for d in args.correctness_dirs]
    keys = sorted(set().union(*runs))

    header = (f"{'condition':<24} {'V':>6} {'C_maj':>6} {'joint':>6} | "
              f"{'C per run':<24} {'joint_mean':>10}")
    cur_cwe = None
    for cwe, mode, cond in keys:
        if cwe != cur_cwe:
            print(f"\n=== {cwe} ===\n{header}")
            cur_cwe = cwe
        per_run = [run.get((cwe, mode, cond), {}) for run in runs]
        ids = set().union(*per_run)
        n = len(ids)
        v = codeql_rates.get(cwe, {}).get(mode, {}).get(cond)
        if v is None:
            print(f"{cond:<24} skipped (no CodeQL result)")
            continue
        V = v["n_vuln"] / n
        # An item missing from a run (e.g. a cancelled batch) is decided by the
        # runs that judged it; a tie counts as incorrect.
        votes = [[p[i] for p in per_run if i in p] for i in ids]
        C_maj = sum(2 * sum(vs) > len(vs) for vs in votes) / n
        C_runs = [sum(p.values()) / len(p) if p else float("nan") for p in per_run]
        joint_mean = sum(c * (1 - V) for c in C_runs) / len(C_runs)
        missing = sum(n - len(p) for p in per_run)
        note = f"  ({missing} missing judgments)" if missing else ""
        print(f"{cond:<24} {100*V:6.1f} {100*C_maj:6.1f} {100*C_maj*(1-V):6.1f} | "
              f"{' '.join(f'{100*c:5.1f}' for c in C_runs):<24} {100*joint_mean:10.1f}{note}")


if __name__ == "__main__":
    main()
