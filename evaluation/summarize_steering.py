"""
Join CodeQL and GPT-4.1 results into V / C / C(1-V) per steering condition.

Both metrics are recomputed with the paper's definitions instead of the raw
evaluation-script outputs:
  V = n_vuln / n_total, where n_total counts every generation (the CodeQL
      script drops syntactically invalid code from its own denominator).
  C = all judged severities are Negligible (App. B.2); Small and unparseable
      judge outputs count as incorrect.

Usage:
  python evaluation/summarize_steering.py \\
      --codeql_rates results/qwen/codeql_steered/detection_rates.json \\
      --correctness_dir results/qwen/correctness/steered \\
      --out_csv results/qwen/steering_summary.csv
"""
import argparse
import csv
import json
import re
from collections import defaultdict

MODE_ORDER = ["baseline", "layer_md", "probe_md", "causal_md"]
SEVERITY_RE = re.compile(r'"severity"\s*:\s*"([^"]+)"')


def strict_correct_counts(labeled_path):
    """(cwe, mode, condition) -> number of generations whose severities are all Negligible."""
    counts = defaultdict(int)
    with open(labeled_path) as f:
        for line in f:
            r = json.loads(line)
            sevs = SEVERITY_RE.findall(r.get("gpt41_raw") or "")
            if sevs and all(s.strip() == "Negligible" for s in sevs):
                counts[(r["cwe_id"], r["mode"], r["condition"])] += 1
    return counts


def build_rows(codeql_rates, correctness_rates, strict):
    rows = []
    for cwe, by_mode in correctness_rates.items():
        for mode, by_cond in by_mode.items():
            for cond, c in by_cond.items():
                v = codeql_rates.get(cwe, {}).get(mode, {}).get(cond)
                if v is None:
                    print(f"  no CodeQL result for {cwe}/{mode}/{cond}, skipped")
                    continue
                n = c["n_total"]
                V = v["n_vuln"] / n
                C = strict[(cwe, mode, cond)] / n
                rows.append({
                    "cwe": cwe, "mode": mode, "condition": cond, "n": n,
                    "n_parsable": v["n_total"], "V": V, "C": C, "joint": C * (1 - V),
                    "V_script": v["vuln_rate"], "C_script": c["correctness_rate"],
                })
    return rows


def fmt(r):
    return (f"{r['mode']:<10} {r['condition']:<36} {100*r['V']:6.1f} {100*r['C']:6.1f} "
            f"{100*r['joint']:7.1f} | {100*r['V_script']:6.1f} {100*r['C_script']:6.1f} "
            f"{r['n_parsable']:>3}/{r['n']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--codeql_rates", required=True,
                    help="detection_rates.json from codeql_eval_steered_modes.py")
    ap.add_argument("--correctness_dir", required=True,
                    help="Directory with correctness_rates.json and labeled.jsonl "
                         "from eval_correctness_steered.py collect")
    ap.add_argument("--out_csv", default=None, help="Optional CSV with every condition")
    ap.add_argument("--all", action="store_true", help="Also print every condition")
    args = ap.parse_args()

    with open(args.codeql_rates) as f:
        codeql_rates = json.load(f)
    with open(f"{args.correctness_dir}/correctness_rates.json") as f:
        correctness_rates = json.load(f)
    strict = strict_correct_counts(f"{args.correctness_dir}/labeled.jsonl")
    rows = build_rows(codeql_rates, correctness_rates, strict)

    header = (f"{'mode':<10} {'config':<36} {'V':>6} {'C':>6} {'C(1-V)':>7} | "
              f"{'V_scr':>6} {'C_scr':>6} {'parse':>7}")
    mode_key = lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else len(MODE_ORDER)
    for cwe in sorted({r["cwe"] for r in rows}):
        cwe_rows = [r for r in rows if r["cwe"] == cwe]
        print(f"\n=== {cwe}: best config per mode by C(1-V) ===\n{header}")
        for mode in sorted({r["mode"] for r in cwe_rows}, key=mode_key):
            print(fmt(max((r for r in cwe_rows if r["mode"] == mode), key=lambda r: r["joint"])))
        if args.all:
            print(f"\n--- {cwe}: all configs ---\n{header}")
            for r in sorted(cwe_rows, key=lambda r: -r["joint"]):
                print(fmt(r))

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["cwe"], mode_key(r["mode"]), -r["joint"])))
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
