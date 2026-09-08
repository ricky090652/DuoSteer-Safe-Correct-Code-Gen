"""
DS-4: CodeQL safety verification on DS-3 filtered outputs.

For each CWE, extracts Python code into per-condition subfolders, runs
CodeQL with the target-CWE query only, and annotates every filtered record
with codeql_pass/fail. Outputs a merged codeql_annotated.jsonl per CWE.

Usage (one run per CWE):
    python codeql_filter.py --cwe cwe-022
    python codeql_filter.py --cwe cwe-022 --dry_run
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
import subprocess
import sys
from pathlib import Path

from codeql_entry_points import add_entry_point

CODEQL_BIN  = os.environ.get("CODEQL_BIN", "codeql")
QLPACK_BASE = os.environ.get("CODEQL_QLPACK", "")

CWE_QUERIES = {
    "cwe-022": "CWE-022",        # directory: PathInjection.ql + TarSlip.ql
    "cwe-079": "CWE-079",        # directory: ReflectedXss.ql + Jinja2WithoutEscaping.ql
    "cwe-094": "CWE-094/CodeInjection.ql",
    "cwe-295": "CWE-295",        # directory: MissingHostKeyValidation.ql + RequestWithoutValidation.ql
    "cwe-502": "CWE-502/UnsafeDeserialization.ql",
}

FILTERED_BASE  = Path("data/double_steering/filtered")
CODEQL_IN_BASE = Path("data/double_steering/codeql_input")
CODEQL_DB_BASE = Path("data/double_steering/codeql_db")
CODEQL_RES_BASE = Path("data/double_steering/codeql_results")


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def safe_filename(record_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", record_id) + ".py"


def extract_code_files(cwe: str) -> dict:
    """Write .py files to CODEQL_IN_BASE/cwe/{config}__{condition}/id.py.
    Returns mapping from file_path -> (record_id, config, condition)."""
    cwe_in = CODEQL_IN_BASE / cwe
    cwe_in.mkdir(parents=True, exist_ok=True)

    path_to_meta = {}
    filt_cwe = FILTERED_BASE / cwe
    if not filt_cwe.exists():
        return path_to_meta

    for config_dir in sorted(filt_cwe.iterdir()):
        if not config_dir.is_dir():
            continue
        config = config_dir.name
        for jsonl_file in sorted(config_dir.glob("*.jsonl")):
            condition = jsonl_file.stem  # filename without .jsonl
            out_dir = cwe_in / f"{config}__{condition}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(jsonl_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("filter_status") != "pass":
                        continue
                    code = rec.get("extracted_code") or extract_code(
                        rec.get("predicted_code", ""))
                    code = add_entry_point(code, cwe)
                    fname = safe_filename(rec["id"])
                    py_path = out_dir / fname
                    py_path.write_text(code)
                    path_to_meta[str(py_path)] = {
                        "id": rec["id"],
                        "config": config,
                        "condition": condition,
                        "alpha": rec.get("steering_info", {}).get("alpha",
                                  rec.get("alpha")),
                    }
    return path_to_meta


def run_codeql(cwe: str, dry_run: bool) -> Path:
    """Create DB and run target-CWE query. Returns SARIF output path."""
    src_root = CODEQL_IN_BASE / cwe
    db_path  = CODEQL_DB_BASE / cwe
    out_dir  = CODEQL_RES_BASE / cwe
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sarif_path = out_dir / "issues_target.sarif"

    if dry_run:
        print(f"[dry_run] Would create DB at {db_path} from {src_root}")
        print(f"[dry_run] Would write SARIF to {sarif_path}")
        return sarif_path

    # Remove old DB if exists
    if db_path.exists():
        subprocess.run(["rm", "-rf", str(db_path)], check=True)

    ql = f"{QLPACK_BASE}/{CWE_QUERIES[cwe]}"
    print(f"Creating CodeQL DB for {cwe}...")
    # Exit code 2 means partial success (some files failed extraction, e.g. due
    # to Python extractor pre-finalize errors on edge-case syntax). The DB is
    # still usable for analysis — treat codes 0 and 2 as success.
    result = subprocess.run([
        CODEQL_BIN, "database", "create", str(db_path),
        "--language", "python",
        "--source-root", str(src_root),
        "--overwrite",
    ])
    if result.returncode not in (0, 2):
        raise subprocess.CalledProcessError(result.returncode, result.args)

    print(f"Analyzing {cwe} with {ql}...")
    result = subprocess.run([
        CODEQL_BIN, "database", "analyze", str(db_path),
        ql,
        "--format", "sarif-latest",
        "--output", str(sarif_path),
        "--rerun",
    ])
    if result.returncode not in (0, 2):
        raise subprocess.CalledProcessError(result.returncode, result.args)

    return sarif_path


def parse_sarif_alerts(sarif_path: Path) -> set:
    """Return set of .py file paths that have at least one alert."""
    if not sarif_path.exists():
        return set()
    with open(sarif_path) as f:
        sarif = json.load(f)
    flagged = set()
    for run in sarif.get("runs", []):
        for result in run.get("results", []):
            for loc in result.get("locations", []):
                uri = loc.get("physicalLocation", {}) \
                         .get("artifactLocation", {}).get("uri", "")
                flagged.add(uri)
    return flagged


def annotate_records(cwe: str, path_to_meta: dict, flagged_files: set) -> list:
    """Merge CodeQL results back into filtered records."""
    # Composite key (config, condition, id) to avoid last-wins overwrite
    # when the same question ID appears in multiple steering conditions.
    id_to_pass = {}
    for path_str, meta in path_to_meta.items():
        py_name = Path(path_str).name
        condition_dir = Path(path_str).parent.name  # config__condition
        # SARIF URIs are relative paths from the DB source root
        rel = f"{condition_dir}/{py_name}"
        passed = rel not in flagged_files
        key = (meta["config"], meta["condition"], meta["id"])
        id_to_pass[key] = passed

    out_path = CODEQL_RES_BASE / cwe / "detection_results.json"
    results = []
    filt_cwe = FILTERED_BASE / cwe
    for config_dir in sorted(filt_cwe.iterdir()):
        if not config_dir.is_dir():
            continue
        config = config_dir.name
        for jsonl_file in sorted(config_dir.glob("*.jsonl")):
            condition = jsonl_file.stem
            with open(jsonl_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("filter_status") != "pass":
                        continue
                    rid = rec["id"]
                    codeql_pass = id_to_pass.get((config, condition, rid), True)
                    results.append({
                        "id": rid,
                        "config": config,
                        "condition": condition,
                        "alpha": rec.get("steering_info", {}).get("alpha",
                                  rec.get("alpha")),
                        "codeql_pass": codeql_pass,
                    })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  detection_results.json: {len(results)} records, "
          f"{sum(r['codeql_pass'] for r in results)} pass")
    return results


def write_annotated_jsonl(cwe: str, results: list):
    """Write codeql_annotated.jsonl by enriching filtered records."""
    id_to_pass = {(r["config"], r["condition"], r["id"]): r["codeql_pass"] for r in results}
    out_path = FILTERED_BASE / cwe / "codeql_annotated.jsonl"
    count_pass = 0
    with open(out_path, "w") as fout:
        filt_cwe = FILTERED_BASE / cwe
        for config_dir in sorted(filt_cwe.iterdir()):
            if not config_dir.is_dir():
                continue
            for jsonl_file in sorted(config_dir.glob("*.jsonl")):
                condition = jsonl_file.stem
                config = config_dir.name
                with open(jsonl_file) as fin:
                    for line in fin:
                        rec = json.loads(line)
                        if rec.get("filter_status") != "pass":
                            continue
                        rid = rec["id"]
                        rec["codeql_pass"] = id_to_pass.get((config, condition, rid), True)
                        rec["config"] = config
                        rec["condition"] = condition
                        if rec["codeql_pass"]:
                            count_pass += 1
                        fout.write(json.dumps(rec) + "\n")
    print(f"  codeql_annotated.jsonl: {count_pass} codeql_pass=True records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwe", required=True,
                        choices=["cwe-022","cwe-079","cwe-094","cwe-295","cwe-502"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--codeql", default=None,
                        help="Path to the CodeQL CLI binary (default: $CODEQL_BIN or 'codeql')")
    parser.add_argument("--qlpack_base", default=None,
                        help="Path to the CodeQL python-queries Security directory (default: $CODEQL_QLPACK)")
    args = parser.parse_args()
    global CODEQL_BIN, QLPACK_BASE
    if args.codeql:
        CODEQL_BIN = args.codeql
    if args.qlpack_base:
        QLPACK_BASE = args.qlpack_base

    cwe = args.cwe

    print(f"=== DS-4 CodeQL filter: {cwe} ===")
    print("Extracting code files...")
    path_to_meta = extract_code_files(cwe)
    print(f"  {len(path_to_meta)} .py files written")

    sarif_path = run_codeql(cwe, args.dry_run)

    if not args.dry_run:
        flagged = parse_sarif_alerts(sarif_path)
        print(f"  {len(flagged)} files flagged by CodeQL")
    else:
        flagged = set()

    results = annotate_records(cwe, path_to_meta, flagged)
    write_annotated_jsonl(cwe, results)
    print(f"=== {cwe} DONE ===")


if __name__ == "__main__":
    main()
