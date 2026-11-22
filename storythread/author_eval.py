"""Eval-scenario helpers for the authoring page (author_ui --kind eval).

An eval scenario file holds rows in the eval/dev.jsonl format that
scenarios.parse_scenarios reads. The page edits one shared history with two
variants of a single turn and builds the two message lists itself.
"""
import hashlib
import json
from pathlib import Path

from .data import rows
from .scenarios import VERSIONS, check_disjoint, parse_scenarios


def strip(value):
    """Returns a stripped string, or value when it is not a string."""
    return value.strip() if isinstance(value, str) else value


def eval_from(payload):
    """Copies eval scenario fields from a request body, stripping text.

    Raises ValueError when the shape is wrong. parse_scenarios does the
    rest of the checking.
    """
    if not isinstance(payload, dict):
        raise ValueError("Expected a scenario object.")
    raw_versions = payload.get("versions")
    if not isinstance(raw_versions, dict):
        raise ValueError("Expected versions a and b.")
    versions = {}
    for name in VERSIONS:
        raw = raw_versions.get(name)
        if not isinstance(raw, dict) or not isinstance(
                raw.get("messages"), list):
            raise ValueError(f"missing version {name}")
        messages = []
        for item in raw["messages"]:
            item = item if isinstance(item, dict) else {}
            messages.append({"role": item.get("role"),
                             "content": strip(item.get("content"))})
        versions[name] = {
            "messages": messages,
            "expected": strip(raw.get("expected")),
            "evidence_turn": raw.get("evidence_turn"),
            "evidence": strip(raw.get("evidence")),
        }
    return {"scenario_id": strip(payload.get("scenario_id")),
            "group": strip(payload.get("group")),
            "versions": versions}


def check_eval(collected, other=None):
    """Raises ValueError unless collected parses and stays disjoint.

    other is the sibling eval file (dev or test). It is skipped when it is
    missing or empty.
    """
    parsed = parse_scenarios(collected)
    if other is not None and Path(other).is_file():
        theirs = list(rows(other))
        if theirs:
            check_disjoint(parsed, parse_scenarios(theirs))


def eval_turns(row):
    """Returns the number of turns after the system card, or 0."""
    try:
        return len(row["versions"]["a"]["messages"]) - 1
    except (KeyError, TypeError):
        return 0


def assert_editable(path, new_text, eval_dir):
    """Raises ValueError when a generation output hashed the current file.

    path is the eval scenario file. new_text is what would replace it.
    eval_dir is the directory `eval run` writes (work/eval by default).
    A byte-for-byte rewrite is allowed. A file that has not been generated
    against is allowed.
    """
    path = Path(path)
    if not path.is_file():
        return
    old = path.read_bytes()
    new = new_text.encode("utf-8")
    if hashlib.sha256(old).digest() == hashlib.sha256(new).digest():
        return
    digest = hashlib.sha256(old).hexdigest()
    folder = Path(eval_dir)
    if not folder.is_dir():
        return
    for candidate in sorted(folder.glob("*.jsonl")):
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(row, dict) and row.get("cases_sha") == digest:
                raise ValueError(
                    f"{path.name} was already generated against. Delete "
                    f"the outputs in {folder} before editing it.")


def sibling(path):
    """Returns eval/test.jsonl for eval/dev.jsonl and back, else None."""
    path = Path(path)
    name = {"dev.jsonl": "test.jsonl", "test.jsonl": "dev.jsonl"}.get(
        path.name)
    return path.with_name(name) if name else None
