"""Local editor for authored/scenes.jsonl or eval/dev.jsonl.

Serves one HTML page on 127.0.0.1. Saves scene pairs in the authored format,
or with --kind eval, two-version eval scenarios. Token checks use the same
measurement as runtime.preflight.

Run: python -m storythread.author_ui
     python -m storythread.author_ui --kind eval
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .author_eval import (
    check_eval, eval_from, eval_turns, sibling)
from .data import rows
from .export import authored_example
from .review_ui import (
    HOST, MAX_BODY, assemble_page, page_paths, read_lines, write_atomic)
from .runtime import DEFAULT_MODEL, flags_for, measure
from .scenarios import BRANCHES, parse_authored

PORT = 8766
HTML = Path(__file__).parent / "web" / "author" / "author_ui.html"
PAGE_STYLE = "author_ui.css"
COMMON_SCRIPTS = (
    "author_ui_core.js",
    "author_ui_shared.js",
    "author_ui_list.js",
)
SCRIPTS = {
    "scenes": COMMON_SCRIPTS + (
        "author_ui_state.js",
        "author_ui_chat.js",
        "author_ui_fields.js",
        "author_ui_pages.js",
        "author_ui_start.js",
    ),
    "eval": COMMON_SCRIPTS + (
        "author_ui_eval_state.js",
        "author_ui_eval_chat.js",
        "author_ui_eval_fields.js",
        "author_ui_start.js",
    ),
}
DEFAULT_FILES = {
    "scenes": "authored/scenes.jsonl",
    "eval": "eval/dev.jsonl",
}


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
                 kind="scenes", other=None):
        """path is the JSONL file. tokenizer skips loading model.

        kind is "scenes" (authored pairs) or "eval" (two-version eval
        scenarios). other is the sibling eval file to stay disjoint from.
        """
        self.path = Path(path)
        self.model = model
        self.kind = kind
        self.other = other
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
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(self.path, text)
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
            write_atomic(self.path, text)
        print(f"deleted {scenario_id}", flush=True)
        return removed

    def tokenizer(self):
        """Returns the cached tokenizer, loading it on first use."""
        if self._tokenizer is not None:
            return self._tokenizer
        with self.lock:
            if self._tokenizer is None:
                self._tokenizer = load_tokenizer(self.model)
        return self._tokenizer

    def token_report(self, payload, cap):
        """Measures both endings the way runtime.preflight measures a row.

        Uses runtime.measure and runtime.flags_for. cap is the training
        token limit. Returns max_tokens plus a recent and earlier report.
        Does not write the scene file.
        """
        if type(cap) is not int or cap < 1:
            raise ValueError("max_tokens must be a positive integer.")
        scene = scene_from(payload)
        require_quotes([scene])
        parsed = parse_authored([scene])[0]
        tokenizer = self.tokenizer()
        report = {"max_tokens": cap}
        for name in BRANCHES:
            example, manifest = authored_example(parsed, name)
            info = measure(
                tokenizer, example["messages"], manifest["evidence_turn"])
            report[name] = {
                "input": info["input"],
                "target": info["target"],
                "full": info["full"],
                "flags": flags_for(info, cap),
                "evidence_distance_tokens": info.get(
                    "evidence_distance_tokens"),
                "distance_turns": manifest["evidence_distance_turns"],
            }
        return report


def make_server(scenes, port, tokenizer=None, model=DEFAULT_MODEL,
                kind="scenes", other=None):
    """Returns a server bound to 127.0.0.1. port 0 picks a free port.

    kind and other are described on Store.
    """
    store = Store(scenes, tokenizer=tokenizer, model=model, kind=kind,
                  other=other)
    scripts = SCRIPTS[kind]

    class Handler(BaseHTTPRequestHandler):
        """Serves the page and the authoring JSON routes."""

        def do_GET(self):
            """Handles the page, the scene list, and one scene."""
            self._route("GET")

        def do_POST(self):
            """Handles create, update, and token checks."""
            self._route("POST")

        def do_DELETE(self):
            """Handles deleting one saved scene."""
            self._route("DELETE")

        def _is_page(self, path):
            """True for the editor shell, including /new and /scene/."""
            if path in ("/", "/index.html", "/new"):
                return True
            return path.startswith("/scene/") and path != "/scene/"

        def _route(self, method):
            """Dispatches one request, or returns a JSON error."""
            path = unquote(urlparse(self.path).path)
            try:
                if method == "GET" and self._is_page(path):
                    page = assemble_page(HTML, PAGE_STYLE, scripts)
                    self._send(200, "text/html; charset=utf-8", page)
                    return
                if method == "GET" and path == "/api/info":
                    self._json(200, {"kind": kind, "file": str(store.path)})
                    return
                if method == "GET" and path == "/api/scenes":
                    self._json(200, store.list_scenes())
                    return
                if method == "POST" and path == "/api/scenes":
                    scene = store.save(None, self._read_json())
                    self._json(200, {"ok": True, "scene": scene})
                    return
                if (method == "POST" and path == "/api/tokens"
                        and kind == "scenes"):
                    payload = self._read_json()
                    report = store.token_report(
                        payload, payload.get("max_tokens", 512))
                    self._json(200, report)
                    return
                prefix = "/api/scenes/"
                if path.startswith(prefix) and len(path) > len(prefix):
                    scenario_id = path[len(prefix):]
                    if method == "GET":
                        self._json(200, store.get_scene(scenario_id))
                        return
                    if method == "POST":
                        scene = store.save(scenario_id, self._read_json())
                        self._json(200, {"ok": True, "scene": scene})
                        return
                    if method == "DELETE":
                        store.delete(scenario_id)
                        self._json(200, {"ok": True})
                        return
                self._json(404, {"error": "Not found."})
            except KeyError:
                self._json(404, {"error": "Unknown scene."})
            except ValueError as error:
                self._json(400, {"error": str(error)})

        def _read_json(self):
            """Returns the POST body as an object."""
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                raise ValueError("Expected a JSON body.")
            try:
                payload = json.loads(
                    self.rfile.read(length).decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                raise ValueError("Expected a JSON object.") from None
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            return payload

        def _json(self, code, payload):
            """Sends payload as JSON."""
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(code, "application/json; charset=utf-8", raw)

        def _send(self, code, content_type, raw):
            """Sends raw bytes with a content type and no cache."""
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt, *args):
            """Drops the default access log; saves print themselves."""
            return

    return ThreadingHTTPServer((HOST, port), Handler)


def main():
    """Serves the authoring page until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(SCRIPTS), default="scenes")
    parser.add_argument(
        "--scenes", help="file to edit (default depends on --kind)")
    parser.add_argument(
        "--other", help="sibling eval file; default swaps dev and test")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    scenes = args.scenes or DEFAULT_FILES[args.kind]
    other = args.other or (sibling(scenes) if args.kind == "eval" else None)
    missing = [
        path.name
        for path in page_paths(HTML, PAGE_STYLE, SCRIPTS[args.kind])
        if not path.is_file()]
    if missing:
        raise SystemExit("Missing " + ", ".join(missing))
    server = make_server(scenes, args.port, model=args.model,
                         kind=args.kind, other=other)
    print(f"Author UI ({args.kind}: {scenes}) at "
          f"http://{HOST}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
