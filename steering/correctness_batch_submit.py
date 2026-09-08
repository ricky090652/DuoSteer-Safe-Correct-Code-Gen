"""
DS-5b: Submit GPT-4.1 batch jobs to OpenAI API, one per CWE.

Usage:
    python correctness_batch_submit.py
    python correctness_batch_submit.py --cwes cwe-022 cwe-079
"""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

LABEL_BASE = Path("data/double_steering/correctness_labels")
CWES = ["cwe-022", "cwe-079", "cwe-094", "cwe-295", "cwe-502"]


def submit_batch(client: OpenAI, cwe: str) -> dict:
    batch_input = LABEL_BASE / cwe / "batch_input_gpt41.jsonl"
    if not batch_input.exists():
        print(f"{cwe}: batch_input_gpt41.jsonl not found — run ds5_prepare first")
        return {}

    print(f"{cwe}: uploading batch file...")
    with open(batch_input, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")

    print(f"{cwe}: creating batch job (file_id={upload.id})...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    state = {
        "cwe":          cwe,
        "batch_id":     batch.id,
        "input_file_id": upload.id,
        "status":       batch.status,
    }
    state_path = LABEL_BASE / cwe / "batch_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"{cwe}: batch_id={batch.id}  status={batch.status}")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwes", nargs="+", default=CWES)
    parser.add_argument("--label_base", default="data/double_steering/correctness_labels",
                        help="Directory with batch_input_gpt41.jsonl per CWE")
    args = parser.parse_args()
    global LABEL_BASE
    LABEL_BASE = Path(args.label_base)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for cwe in args.cwes:
        submit_batch(client, cwe)


if __name__ == "__main__":
    main()
