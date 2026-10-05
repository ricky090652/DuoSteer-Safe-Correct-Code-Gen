"""Thin JSON/JSONL I/O wrappers and shared helpers."""
import hashlib
import json


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def question_hash(question: str) -> str:
    """Stable id derived from the question text."""
    return "q_" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def question_group_id(pair: dict) -> str:
    """
    Question-level grouping key (train/val splits, task deduplication).

    The released pair files carry no `src_id`, and one question backs many
    pairs, so falling back to the per-pair `id` would leak questions across
    splits. Use `src_id` when present, else a hash of the question text.
    """
    if pair.get("src_id"):
        return str(pair["src_id"])
    return question_hash(pair["question"])
