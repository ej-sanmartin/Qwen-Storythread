"""Local reviewer for work/review.jsonl.

Serves one HTML page on 127.0.0.1. Decisions are written back to the review
file; data.py prepare still selects rows with approved true.

Run: python -m storythread.review_ui
"""
import argparse
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .data import rows

HOST = "127.0.0.1"
PORT = 8765
STATUSES = ("pending", "skipped", "approved")
MAX_BODY = 2_000_000
HTML = Path(__file__).parent / "web" / "review" / "review_ui.html"
PAGE_STYLE = "review_ui.css"
PAGE_SCRIPTS = ("review_ui_queue.js", "review_ui_scene.js")
SUGGESTION_KEYS = (
    "judge_model", "status", "total", "start", "end", "keep_reason",
    "concern", "skip_reason", "flags", "name")


def page_paths(shell, style, scripts):
    """Returns the shell, stylesheet, and script paths for one page."""
    root = Path(shell).parent
    return [Path(shell), root / style, *(root / name for name in scripts)]


def assemble_page(shell, style, scripts):
    """Returns one HTML page with the stylesheet and scripts inserted.

    shell is the HTML path. style and scripts are filenames beside it.
    {{style}} and {{script}} in the shell are replaced. Reads every call
    so a refresh shows edits without restarting the server.
    """
    root = Path(shell).parent
    text = Path(shell).read_text(encoding="utf-8")
    css = (root / style).read_text(encoding="utf-8").rstrip("\n")
    blocks = [
        (root / name).read_text(encoding="utf-8").rstrip("\n")
        for name in scripts]
    if "{{style}}" not in text or "{{script}}" not in text:
        raise ValueError(f"{Path(shell).name} is missing a page placeholder")
    text = text.replace("{{style}}", css, 1)
    text = text.replace("{{script}}", "\n".join(blocks), 1)
    return text.encode("utf-8")


def status_of(row):
    """Returns pending, skipped, or approved for a review row.

    Rows written before review_status existed map approved true to approved
    and every other value to pending.
    """
    status = row.get("review_status")
    if status in STATUSES:
        return status
    if row.get("approved") is True:
        return "approved"
    return "pending"


def primary_suggestion(items):
    """Returns the latest suggestion that has a score, else the latest."""
    scored = [item for item in items if type(item.get("total")) is int]
    if scored:
        return scored[-1]
    if items:
        return items[-1]
    return None


def validate_window(turns, start, end):
    """Raises ValueError unless [start, end) ends user, then assistant.

    end is exclusive, matching review.jsonl and data.py prepare.
    """
    count = len(turns)
    if not 0 <= start < end <= count:
        raise ValueError(
            f"Range {start}:{end} is outside 0:{count} (end is exclusive).")
    user_then_assistant = (
        end - start >= 2
        and turns[end - 2].get("role") == "user"
        and turns[end - 1].get("role") == "assistant")
    if not user_then_assistant:
        raise ValueError(
            "Approved window must end with a user turn "
            "followed by an assistant reply.")


def suggestion_view(row):
    """Copies the suggestion fields the page displays."""
    return {key: row[key] for key in SUGGESTION_KEYS if key in row}


def read_lines(path):
    """Returns (lines, newline, trailing) without changing other rows."""
    raw = Path(path).read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8")
    return text.splitlines(), newline, text.endswith("\n")


def write_atomic(path, text):
    """Replaces path with text via a temp file in the same directory."""
    path = Path(path)
    mode = path.stat().st_mode if path.exists() else 0o644
    fd, name = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(tmp, mode & 0o777)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_scenes(index_path, review_path):
    """Loads name and turns for scene ids that appear in the review file."""
    wanted = set()
    if Path(review_path).is_file():
        for row in rows(review_path):
            if row.get("scene_id"):
                wanted.add(row["scene_id"])
    found = {}
    if not wanted or not Path(index_path).is_file():
        return found
    for row in rows(index_path):
        scene_id = row.get("scene_id")
        if scene_id in wanted and scene_id not in found:
            found[scene_id] = {
                "name": row.get("name") or "",
                "turns": row.get("turns") or [],
            }
            if len(found) == len(wanted):
                break
    return found


class Store:
    """Reads the index and suggestions, and writes one review row at a time."""

    def __init__(self, review, index, suggestions):
        self.review_path = Path(review)
        self.index_path = Path(index)
        self.suggestions_path = Path(suggestions)
        self.lock = threading.Lock()
        self._scenes = None

    def scenes(self):
        """Returns cached index scenes for the current review ids."""
        if self._scenes is None:
            with self.lock:
                if self._scenes is None:
                    self._scenes = load_scenes(
                        self.index_path, self.review_path)
        return self._scenes

    def review_rows(self):
        """Returns every review row, in file order."""
        if not self.review_path.is_file():
            return []
        return list(rows(self.review_path))

    def suggestions(self):
        """Groups suggestion rows by scene_id, in file order."""
        grouped = {}
        if not self.suggestions_path.is_file():
            return grouped
        for row in rows(self.suggestions_path):
            scene_id = row.get("scene_id")
            if scene_id:
                grouped.setdefault(scene_id, []).append(row)
        return grouped

    def turns(self, scene_id):
        """Returns indexed turns, or None when the scene is absent."""
        scene = self.scenes().get(scene_id)
        if scene is None:
            return None
        return scene.get("turns") or []

    def queue(self):
        """Returns the left-pane rows, including a score when one exists."""
        scenes = self.scenes()
        suggestions = self.suggestions()
        items = []
        for order, row in enumerate(self.review_rows()):
            scene_id = row.get("scene_id")
            packed = suggestions.get(scene_id, [])
            primary = primary_suggestion(packed) or {}
            scene = scenes.get(scene_id, {})
            score = primary.get("total")
            start, end = primary.get("start"), primary.get("end")
            items.append({
                "order": order,
                "scene_id": scene_id,
                "name": scene.get("name") or primary.get("name") or "",
                "status": status_of(row),
                "score": score if type(score) is int else None,
                "suggested_start": start if type(start) is int else None,
                "suggested_end": end if type(end) is int else None,
                "judge_model": primary.get("judge_model") or "",
                "suggestion_count": len(packed),
            })
        return items

    def scene_view(self, scene_id):
        """Returns one scene, its window, and any judge suggestions."""
        row = next((item for item in self.review_rows()
                    if item.get("scene_id") == scene_id), None)
        if row is None:
            raise KeyError(scene_id)
        scene = self.scenes().get(scene_id, {})
        turns = scene.get("turns") or []
        packed = self.suggestions().get(scene_id, [])
        system = row.get("system")
        if not isinstance(system, str):
            system = ""
        return {
            "scene_id": scene_id,
            "name": scene.get("name") or "",
            "status": status_of(row),
            "approved": row.get("approved") is True,
            "group": row.get("group") or "",
            "system": system,
            "start": row.get("start"),
            "end": row.get("end"),
            "turns": [{"role": turn.get("role") or "",
                       "content": turn.get("content") or ""}
                      for turn in turns],
            "suggestions": [suggestion_view(item) for item in packed],
        }

    def decide(self, scene_id, system, start, end, status):
        """Saves system, start, end, review_status, and approved.

        approved is true only when status is approved. Other rows are copied
        unchanged. Approve checks the window before anything is written.
        """
        if status not in STATUSES:
            raise ValueError("status must be pending, skipped, or approved.")
        if not isinstance(system, str):
            raise ValueError("system must be a string.")
        if type(start) is not int or type(end) is not int:
            raise ValueError("start and end must be integers.")
        if status == "approved":
            turns = self.turns(scene_id)
            if turns is None:
                raise ValueError("Scene is not in the index.")
            validate_window(turns, start, end)
        with self.lock:
            lines, newline, trailing = read_lines(self.review_path)
            found = False
            updated = []
            for line in lines:
                if not line.strip():
                    updated.append(line)
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        "Review file has a line that is not JSON.") from None
                if row.get("scene_id") != scene_id:
                    updated.append(line)
                    continue
                row["system"] = system
                row["start"] = start
                row["end"] = end
                row["review_status"] = status
                row["approved"] = status == "approved"
                updated.append(json.dumps(row, ensure_ascii=False))
                found = True
            if not found:
                raise KeyError(scene_id)
            text = newline.join(updated)
            if trailing:
                text += newline
            write_atomic(self.review_path, text)
        print(f"{status} {scene_id}", flush=True)


def make_server(review, index, suggestions, port):
    """Returns a server bound to 127.0.0.1. port 0 picks a free port."""
    store = Store(review, index, suggestions)

    class Handler(BaseHTTPRequestHandler):
        """Serves the page and the review JSON routes."""

        def do_GET(self):
            """Handles queue, scene, and the HTML page."""
            self._route("GET")

        def do_POST(self):
            """Handles a decision for one scene."""
            self._route("POST")

        def _is_page(self, path):
            """True for the queue and for /scene/<scene_id>."""
            if path in ("/", "/index.html"):
                return True
            return path.startswith("/scene/") and path != "/scene/"

        def _route(self, method):
            """Dispatches one request, or returns a JSON error."""
            path = unquote(urlparse(self.path).path)
            try:
                if method == "GET" and self._is_page(path):
                    page = assemble_page(HTML, PAGE_STYLE, PAGE_SCRIPTS)
                    self._send(200, "text/html; charset=utf-8", page)
                    return
                if method == "GET" and path == "/api/queue":
                    self._json(200, store.queue())
                    return
                prefix = "/api/scene/"
                if path.startswith(prefix) and len(path) > len(prefix):
                    scene_id = path[len(prefix):]
                    if method == "GET":
                        self._json(200, store.scene_view(scene_id))
                        return
                    if method == "POST":
                        payload = self._read_json()
                        store.decide(
                            scene_id, payload.get("system"),
                            payload.get("start"), payload.get("end"),
                            payload.get("review_status"))
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
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
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
            """Drops the default access log; decisions print themselves."""
            return

    return ThreadingHTTPServer((HOST, port), Handler)


def main():
    """Serves the reviewer until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", default="work/review.jsonl")
    parser.add_argument("--index", default="work/index.jsonl")
    parser.add_argument(
        "--suggestions", default="work/review_suggestions.jsonl")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if not Path(args.review).is_file():
        raise SystemExit(f"Review file not found: {args.review}")
    if not Path(args.index).is_file():
        raise SystemExit(f"Index file not found: {args.index}")
    missing = [
        path.name for path in page_paths(HTML, PAGE_STYLE, PAGE_SCRIPTS)
        if not path.is_file()]
    if missing:
        raise SystemExit("Missing " + ", ".join(missing))
    if not Path(args.suggestions).is_file():
        print(f"No suggestions at {args.suggestions}; scores will be blank.",
              flush=True)
    server = make_server(
        args.review, args.index, args.suggestions, args.port)
    print(f"Review UI at http://{HOST}:{server.server_address[1]}/",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
