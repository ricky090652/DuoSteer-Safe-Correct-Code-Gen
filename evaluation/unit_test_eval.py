"""Lightweight execution-based (unit-test) correctness evaluation.

Runs SecCodePLT unit tests against existing generation files for the CWEs that
ship tests. No model inference. Each task runs in a hardened subprocess (see
unit_test_runner.py). Results -> a single standalone JSON + a Markdown summary.

Generation files are supplied as MODEL:CWE:ARM=PATH specs, e.g.

    python unit_test_eval.py \
        --parquet /path/to/SecCodePLT/insecure_coding-00000-of-00001.parquet \
        --gen llama:079:baseline=out/cwe-079/baseline.jsonl \
        --gen llama:079:duosteer=out/cwe-079/A_s2.0_stk32_c3.0_ctk32.jsonl

Usage notes:
    python unit_test_eval.py --selftest      # verify sandbox blocks network, then exit
    python unit_test_eval.py ... --limit 2   # dry run: 2 tasks per file
"""
import sys, os, json, subprocess, argparse, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "unit_test_runner.py"

# Filled in from CLI args in main().
PARQUET = None
OUT_DIR = HERE.parent / "results" / "unit_test_eval"
GEN_FILES = {}
WALL_TIMEOUT = 20  # seconds per task


def load_tasks():
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    tasks = {}  # src_id(str) -> {function_name, setup, testcases}
    for _, r in df.iterrows():
        u = r["unittest"]
        tasks[str(r["id"])] = {
            "function_name": r["task_description"].get("function_name", ""),
            "setup": u.get("setup", "") or "",
            "testcases": u.get("testcases", "") or "",
        }
    return tasks


def run_task(job):
    try:
        p = subprocess.run([sys.executable, str(RUNNER)],
                           input=json.dumps(job), capture_output=True,
                           text=True, timeout=WALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {**{k: job[k] for k in ("id", "cwe", "arm")},
                "capability": [], "safety": [], "status": "timeout", "detail": ""}
    out = (p.stdout or "").strip().splitlines()
    if not out:
        return {**{k: job[k] for k in ("id", "cwe", "arm")},
                "capability": [], "safety": [], "status": "crash",
                "detail": (p.stderr or "")[:200]}
    return json.loads(out[-1])


def selftest():
    """Feed the runner a generation that tries to open a socket and call os.system;
    verify both are neutralized. No generation data touched."""
    job = {
        "id": "selftest", "cwe": "000", "arm": "selftest",
        "function_name": "f", "setup": "",
        "predicted_code": (
            "import socket, os\n"
            "def f(x=None):\n"
            "    try:\n"
            "        socket.create_connection(('8.8.8.8', 53), timeout=2)\n"
            "        return 'NET_OPEN'\n"
            "    except Exception:\n"
            "        os.system('echo SHOULD_NOT_PRINT')\n"
            "        return 'BLOCKED'\n"
        ),
        "testcases": "testcases = {'capability': [({}, 'BLOCKED')], 'safety': []}",
    }
    r = run_task(job)
    ok = r.get("capability") == ["pass"]
    print("SANDBOX SELFTEST:", "PASS (network blocked, os.system inert)" if ok else f"FAIL -> {r}")
    return ok


def main():
    global PARQUET, OUT_DIR, GEN_FILES, WALL_TIMEOUT
    ap = argparse.ArgumentParser(
        description="Execution-based (unit-test) correctness evaluation over "
                    "existing generation files, using SecCodePLT test suites.")
    ap.add_argument("--selftest", action="store_true",
                    help="Verify the sandbox blocks network and shell, then exit")
    ap.add_argument("--limit", type=int, default=0, help="tasks per file (0 = all)")
    ap.add_argument("--parquet", default=None,
                    help="Path to the SecCodePLT insecure_coding parquet file "
                         "(download from the Virtue-AI-HUB/SecCodePLT dataset on Hugging Face)")
    ap.add_argument("--gen", action="append", default=[], metavar="MODEL:CWE:ARM=PATH",
                    help="Generation file spec, repeatable. Example: "
                         "llama:079:baseline=out/cwe-079/baseline.jsonl")
    ap.add_argument("--out_dir", default=str(OUT_DIR),
                    help="Output directory for results JSON and Markdown summary")
    ap.add_argument("--timeout", type=int, default=20, help="seconds per task")
    args = ap.parse_args()

    if not selftest():
        print("Aborting: sandbox self-test failed."); sys.exit(1)
    if args.selftest:
        return

    if not args.parquet or not args.gen:
        ap.error("--parquet and at least one --gen spec are required (unless --selftest)")
    PARQUET = Path(args.parquet)
    OUT_DIR = Path(args.out_dir)
    WALL_TIMEOUT = args.timeout
    GEN_FILES = {}
    for spec in args.gen:
        try:
            key, path = spec.split("=", 1)
            model, cwe, arm = key.split(":")
        except ValueError:
            ap.error(f"Bad --gen spec {spec!r}; expected MODEL:CWE:ARM=PATH")
        GEN_FILES[(model, cwe, arm)] = path

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    all_records = []
    t0 = time.time()

    for (model, cwe, arm), rel in GEN_FILES.items():
        path = Path(rel)
        gens = [json.loads(l) for l in open(path)]
        if args.limit:
            gens = gens[:args.limit]
        for g in gens:
            meta = tasks.get(str(g.get("src_id")))
            base = {"id": g["id"], "model": model, "cwe": cwe, "arm": arm}
            if not meta or not meta["testcases"]:
                all_records.append({**base, "capability": [], "safety": [],
                                    "status": "no_testcases", "detail": ""})
                continue
            job = {**base, "predicted_code": g.get("predicted_code", ""),
                   "function_name": meta["function_name"],
                   "setup": meta["setup"], "testcases": meta["testcases"]}
            r = run_task(job); r["model"] = model
            all_records.append(r)
        print(f"  done {model}/{cwe}/{arm}: {len(gens)} tasks")

    out_json = OUT_DIR / "unit_test_results.json"
    json.dump(all_records, open(out_json, "w"), indent=1)
    write_summary(all_records, OUT_DIR / "unit_test_SUMMARY.md",
                  elapsed=time.time() - t0)
    print(f"\nWrote {out_json}  and  unit_test_SUMMARY.md  ({len(all_records)} records)")


def _rate(records, model, cwe, arm, kind, caseidx=None):
    """Task-level pass rate. caseidx=None -> all cases must pass; else that case only."""
    rows = [r for r in records if r.get("model") == model and r["cwe"] == cwe
            and r["arm"] == arm and r["status"] == "ok" and r.get(kind)]
    if caseidx is not None:
        rows = [r for r in rows if len(r[kind]) > caseidx]
        if not rows:
            return None, 0
        passed = sum(1 for r in rows if r[kind][caseidx] == "pass")
    else:
        if not rows:
            return None, 0
        passed = sum(1 for r in rows if all(c == "pass" for c in r[kind]))
    return 100.0 * passed / len(rows), len(rows)


def _tbl(records, model, kind, caseidx=None):
    L = ["| CWE | baseline | DuoSteer | delta |", "|-----|----------|----------|-------|"]
    for cwe in sorted({r["cwe"] for r in records if r.get("model") == model}):
        b, nb = _rate(records, model, cwe, "baseline", kind, caseidx)
        d, nd = _rate(records, model, cwe, "duosteer", kind, caseidx)
        if b is None or d is None:
            L.append(f"| {cwe} | n/a | n/a | |")
        else:
            L.append(f"| {cwe} | {b:.1f} (n={nb}) | {d:.1f} (n={nd}) | {d-b:+.1f} |")
    return L


def write_summary(records, path, elapsed):
    from collections import Counter
    L = ["# Unit-Test Correctness Evaluation - Results", "",
         f"Execution-based evaluation on existing generations (no new inference). Runtime {elapsed:.0f}s. "
         "Per-question SecCodePLT tests. Capability = functional correctness (input-output match); "
         "safety = unsafe inputs are rejected. Task-level pass rate = fraction of tasks passing the stated case(s)."]
    models = sorted({r.get("model") for r in records if r.get("model")})
    cwes = sorted({r["cwe"] for r in records})
    arms = sorted({r["arm"] for r in records})
    for model in models:
        L += ["", f"## {model}", "",
              "### Capability - primary case (first functional test)"] + _tbl(records, model, "capability", 0)
        L += ["", "### Capability - all cases must pass (strict)"] + _tbl(records, model, "capability", None)
        L += ["", "### Safety - unsafe-input rejection"] + _tbl(records, model, "safety", None)
    L += ["", "## Execution status breakdown", "",
          "| model | CWE | arm | ok | gen_error | no_function | timeout | crash | no_testcases |",
          "|-------|-----|-----|----|-----------|-------------|---------|-------|--------------|"]
    for model in models:
        for cwe in cwes:
            for arm in arms:
                c = Counter(r["status"] for r in records
                            if r.get("model") == model and r["cwe"] == cwe and r["arm"] == arm)
                L.append(f"| {model} | {cwe} | {arm} | {c['ok']} | {c['gen_error']} | {c['no_function']} "
                         f"| {c['timeout']} | {c['crash']} | {c['no_testcases']} |")
    L += ["", "_Sandbox: network blocked (verified at startup), os.system/subprocess inert, "
          "CPU/mem rlimits, temp cwd._"]
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
