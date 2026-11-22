"""Reads and writes one authored or eval scenario file.

Used by author_ui. The page module stays the server.
"""
import json
import threading
from pathlib import Path

from .author_eval import (
    assert_editable, check_eval, eval_from, eval_turns)
from .data import rows
from .http_ui import read_lines, write_atomic
from .runtime import DEFAULT_MODEL
from .scenarios import BRANCHES, parse_authored


def as_text(value):
    """Returns a stripped string, or value when it is not a string."""
    if isinstance(value, str):
        return value.strip()
    return value


def dumps(row):
    """Returns one JSON object encoded as a JSONL line without newline."""
    return json.dumps(row, ensure_ascii=False)


def decode_row(line):
    """Returns one JSON object from a JSONL line, or raises ValueError."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        raise ValueError(
            "Authored file has a line that is not JSON.") from None
    if not isinstance(row, dict):
        raise ValueError(
            "Authored file has a line that is not an object.")
    return row


def scene_from(payload):
    """Copies authored fields from a request body.

    Text fields are stripped the same way parse_authored strips them.
    The evidence quote is kept verbatim. Raises ValueError on a bad shape.
    """
    if not isinstance(payload, dict):
        raise ValueError("Expected a scene object.")
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("history: expected a nonempty list of turns")
    turns = []
    for number, item in enumerate(history):
        if not isinstance(item, dict):
            raise ValueError(f"history[{number}]: expected a turn")
        turns.append({
            "role": item.get("role"),
            "content": as_text(item.get("content")),
        })
    raw_branches = payload.get("branches")
    if not isinstance(raw_branches, dict):
        raise ValueError("Expected recent and earlier branches.")
    branches = {}
    for name in BRANCHES:
        raw = raw_branches.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"missing branch {name}")
        branches[name] = {
            "user": as_text(raw.get("user")),
            "reply": as_text(raw.get("reply")),
            "evidence_turn": raw.get("evidence_turn"),
            "evidence": raw.get("evidence"),
        }
    return {
        "scenario_id": as_text(payload.get("scenario_id")),
        "group": as_text(payload.get("group")),
        "system": as_text(payload.get("system")),
        "history": turns,
        "branches": branches,
    }


def require_quotes(scenes):
    """Raises unless every ending has a nonempty evidence quote."""
    for scene in scenes:
        sid = scene.get("scenario_id")
        if isinstance(sid, str) and sid.strip():
            label = sid.strip()
        else:
            label = "scene"
        branches = scene.get("branches") or {}
        for name in BRANCHES:
            raw = branches.get(name) or {}
            quote = raw.get("evidence") if isinstance(raw, dict) else None
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError(
                    f"{label} {name}: evidence quote must be nonempty")


def load_tokenizer(model):
    """Returns a tokenizer for model, or raises ValueError."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model)
    except Exception as error:
        raise ValueError(
            f"Could not load the tokenizer at {model}: {error}") from None


class Store:
    """Reads and writes authored scenes, one saved row at a time."""

    def __init__(self, path, tokenizer=None, model=DEFAULT_MODEL,
                 kind="scenes", other=None, eval_dir="work/eval"):
        """path is the JSONL file. tokenizer skips loading model.

        kind is "scenes" or "eval". other is the sibling eval file.
        eval_dir is where eval run writes; a generated eval file is locked.
        """
        self.path = Path(path)
        self.model = model
        self.kind = kind
        self.other = other
        self.eval_dir = eval_dir
        self._tokenizer = tokenizer
        self.lock = threading.Lock()

    def _clean(self, payload):
        """Returns the stripped row for a request body."""
        if self.kind == "eval":
            return eval_from(payload)
        return scene_from(payload)

    def _check(self, collected):
        """Raises ValueError unless every row would be valid together."""
        if self.kind == "eval":
            check_eval(collected, self.other)
            return
        require_quotes(collected)
        parse_authored(collected)

    def _lines(self):
        """Returns (lines, newline, trailing) for the scene file."""
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return [], "\n", True
        return read_lines(self.path)

    def list_scenes(self):
        """Returns scenario_id, group, and turn count per row, in file order."""
        with self.lock:
            if not self.path.is_file():
                return []
            try:
                loaded = list(rows(self.path))
            except json.JSONDecodeError:
                raise ValueError(
                    "Authored file has a line that is not JSON.") from None
        items = []
        for order, row in enumerate(loaded):
            if not isinstance(row, dict):
                raise ValueError(
                    "Authored file has a line that is not an object.")
            turns = row.get("history")
            count = len(turns) if isinstance(turns, list) else 0
            items.append({
                "order": order,
                "scenario_id": row.get("scenario_id") or "",
                "group": row.get("group") or "",
                "turns": eval_turns(row) if self.kind == "eval" else count,
            })
        return items

    def get_scene(self, scenario_id):
        """Returns one stored row, or raises KeyError."""
        with self.lock:
            if not self.path.is_file():
                raise KeyError(scenario_id)
            try:
                loaded = list(rows(self.path))
            except json.JSONDecodeError:
                raise ValueError(
                    "Authored file has a line that is not JSON.") from None
        for row in loaded:
            if isinstance(row, dict) and row.get("scenario_id") == scenario_id:
                return row
        raise KeyError(scenario_id)

    def save(self, original_id, payload):
        """Inserts or replaces one scene. Other lines stay unchanged.

        original_id is the scenario_id already in the file, or None to
        append. Returns the row written. Raises KeyError if the id is
        missing, or ValueError if the file would be invalid.
        """
        scene = self._clean(payload)
        with self.lock:
            lines, newline, trailing = self._lines()
            written, collected = [], []
            found = False
            stored = scene
            for line in lines:
                if not line.strip():
                    written.append(line)
                    continue
                current = decode_row(line)
                replace = (
                    original_id is not None and not found
                    and current.get("scenario_id") == original_id)
                if not replace:
                    written.append(line)
                    collected.append(current)
                    continue
                stored = dict(current)
                stored.update(scene)
                written.append(dumps(stored))
                collected.append(stored)
                found = True
            if original_id is None:
                written.append(dumps(scene))
                collected.append(scene)
                stored = scene
            elif not found:
                raise KeyError(original_id)
            self._check(collected)
            text = newline.join(written)
            if trailing:
                text += newline
            self._write(text)
        print(f"saved {stored.get('scenario_id')}", flush=True)
        return stored

    def delete(self, scenario_id):
        """Removes one scene. Other lines stay unchanged.

        scenario_id is the id already in the file. Returns the removed
        row. Raises KeyError if that id is missing.
        """
        with self.lock:
            lines, newline, trailing = self._lines()
            written, collected, removed = [], [], None
            for line in lines:
                if not line.strip():
                    written.append(line)
                    continue
                current = decode_row(line)
                if (removed is None
                        and current.get("scenario_id") == scenario_id):
                    removed = current
                    continue
                written.append(line)
                collected.append(current)
            if removed is None:
                raise KeyError(scenario_id)
            if collected:
                self._check(collected)
            text = newline.join(written)
            if written and trailing:
                text += newline
            self._write(text)
        print(f"deleted {scenario_id}", flush=True)
        return removed

    def _write(self, text):
        """Writes text. An eval file already generated against stays put."""
        if self.kind == "eval":
            assert_editable(self.path, text, self.eval_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(self.path, text)

    def tokenizer(self):
        """Returns the cached tokenizer, loading it on first use."""
        if self._tokenizer is not None:
            return self._tokenizer
        with self.lock:
            if self._tokenizer is None:
                self._tokenizer = load_tokenizer(self.model)
        return self._tokenizer
