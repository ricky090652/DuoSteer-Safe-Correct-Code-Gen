"""
DS-5c: Poll batch status, download results, parse GPT-4.1 responses,
and write labeled.jsonl per CWE.

Usage:
    python correctness_batch_collect.py             # check all CWEs
    python correctness_batch_collect.py --cwes cwe-022
"""

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI

LABEL_BASE    = Path("data/double_steering/correctness_labels")
FILTERED_BASE = Path("data/double_steering/filtered")
CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]

SEVERITY_ORDER = {"Negligible": 0, "Small": 1, "Major": 2, "Fatal": 3}


def is_correct(judgments: list) -> bool:
    # Any Fatal/Major inconsistency → incorrect.
    # Also catches cases where GPT puts an inconsistency label in the severity
    # field (e.g. "Logic error") — treat any unrecognised severity as Major.
    for j in judgments:
        sev = j.get("severity", "").strip()
        if sev in ("Fatal", "Major"):
            return False
        if sev not in ("", "None", "Negligible", "Small"):
            return False  # unrecognised → treat as incorrect
    return True


def parse_gpt_response(content: str) -> tuple[bool, str, str]:
    """Returns (correct, severity, raw_response).
    Prompt instructs GPT-4.1 to return a JSON list, so try JSON first."""
    judgments = []

    # Primary: parse JSON list output as instructed by prompt
    try:
        # GPT may wrap in ```json``` fences
        text = content.strip()
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        raw_json = m.group(1) if m else text
        # Use raw_decode to parse exactly the first JSON array and ignore
        # any trailing explanation text GPT appends after the closing ].
        start = raw_json.find("[")
        if start >= 0:
            parsed, _ = json.JSONDecoder().raw_decode(raw_json, start)
        else:
            parsed = json.loads(raw_json)
        if isinstance(parsed, list) and parsed:
            judgments = [
                {"inconsistency": str(j.get("inconsistency", "None")),
                 "severity": str(j.get("severity", "Negligible"))}
                for j in parsed if isinstance(j, dict)
            ]
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    # Fallback: plain-text key:value format
    if not judgments:
        for block in re.split(r"\n\s*\n", content):
            inc_m = re.search(r"Inconsistency:\s*(.+)", block, re.I)
            sev_m = re.search(r"Severity:\s*(.+)", block, re.I)
            if inc_m:
                inc = inc_m.group(1).strip()
                sev = sev_m.group(1).strip() if sev_m else "Negligible"
                judgments.append({"inconsistency": inc, "severity": sev})

    if not judgments:
        return True, "Negligible", content

    correct = is_correct(judgments)
    severity = max((j.get("severity", "Negligible") for j in judgments),
                   key=lambda s: SEVERITY_ORDER.get(s, 0))
    return correct, severity, content


def parse_custom_id(custom_id: str) -> dict:
    """Decode: cwe__config__condition__id"""
    parts = custom_id.split("__", 3)
    if len(parts) == 4:
        return {"cwe_id": parts[0], "config": parts[1],
                "condition": parts[2], "id": parts[3]}
    return {"cwe_id": "", "config": "", "condition": "", "id": custom_id}


def check_and_collect(client: OpenAI, cwe: str) -> bool:
    state_path = LABEL_BASE / cwe / "batch_state.json"
    if not state_path.exists():
        print(f"{cwe}: no batch_state.json — not submitted yet")
        return False

    with open(state_path) as f:
        state = json.load(f)

    batch = client.batches.retrieve(state["batch_id"])
    print(f"{cwe}: batch_id={batch.id}  status={batch.status}  "
          f"completed={batch.request_counts.completed}/"
          f"{batch.request_counts.total}")

    # Update saved status
    state["status"] = batch.status
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    if batch.status != "completed":
        return False

    if batch.output_file_id is None:
        print(f"{cwe}: batch completed but no output file")
        return False

    # Download output
    raw_bytes = client.files.content(batch.output_file_id).content
    lines = raw_bytes.decode().strip().split("\n")
    print(f"{cwe}: downloading {len(lines)} results...")

    # Build (config, condition, id) -> original record from codeql_annotated.jsonl.
    # Keying by id alone would overwrite all per-condition records for the same
    # question, causing every condition to share one predicted_code after dedup.
    ann_path = FILTERED_BASE / cwe / "codeql_annotated.jsonl"
    id_to_rec: dict[tuple, dict] = {}
    if ann_path.exists():
        with open(ann_path) as f:
            for line in f:
                r = json.loads(line)
                key = (r.get("config", ""), r.get("condition", ""), r["id"])
                id_to_rec[key] = r

    out_path = LABEL_BASE / cwe / "labeled.jsonl"
    count = 0
    with open(out_path, "w") as fout:
        for line in lines:
            if not line.strip():
                continue
            resp = json.loads(line)
            meta = parse_custom_id(resp.get("custom_id", ""))
            rid = meta["id"]
            body = resp.get("response", {}).get("body", {})
            content = ""
            if body.get("choices"):
                content = body["choices"][0].get("message", {}).get("content", "")
            correct, severity, raw = parse_gpt_response(content)

            key = (meta["config"], meta["condition"], rid)
            orig = id_to_rec.get(key) or id_to_rec.get(("", "", rid), {})
            labeled_rec = {
                "id":           rid,
                "cwe_id":       meta["cwe_id"] or orig.get("cwe_id", cwe),
                "question":     orig.get("question", ""),
                "predicted_code": orig.get("predicted_code", ""),
                "messages":     orig.get("messages", []),
                "config":       meta["config"] or orig.get("config", ""),
                "condition":    meta["condition"] or orig.get("condition", ""),
                "alpha":        orig.get("alpha") or orig.get(
                                    "steering_info", {}).get("alpha"),
                "src_id":       orig.get("src_id", rid),
                "codeql_pass":  True,
                "gpt41_correct": correct,
                "gpt41_severity": severity,
                "gpt41_raw":    raw,
            }
            fout.write(json.dumps(labeled_rec) + "\n")
            count += 1

    print(f"{cwe}: {count} labeled records written to {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwes", nargs="+", default=CWES)
    parser.add_argument("--filtered_base", default="data/double_steering/filtered",
                        help="Directory with CodeQL-annotated filtered generations")
    parser.add_argument("--label_base", default="data/double_steering/correctness_labels",
                        help="Directory for batch files and correctness labels")
    args = parser.parse_args()
    global FILTERED_BASE, LABEL_BASE
    FILTERED_BASE = Path(args.filtered_base)
    LABEL_BASE = Path(args.label_base)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for cwe in args.cwes:
        check_and_collect(client, cwe)


if __name__ == "__main__":
    main()
