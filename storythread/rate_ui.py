"""Local page for rating blinded eval replies.

Reads the sheets that `eval run` writes (continuity sheet and engagement
pairs) and saves each rating straight back to its file. The key files stay
on disk: the page never learns which model wrote which reply.

Run: python -m storythread.rate_ui [--dir work/eval]
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .data import rows
from .eval_flow import DEFAULT_DIR, sheets
from .review_ui import (
    HOST, MAX_BODY, assemble_page, page_paths, write_atomic)

PORT = 8769
HTML = Path(__file__).parent / "web" / "rate" / "rate_ui.html"
PAGE_STYLE = "rate_ui.css"
PAGE_SCRIPTS = ("rate_ui.js",)
WINNERS = ("A", "B", "tie", "")


class Store:
    """Reads sheets and saves one case's rating at a time."""

    def __init__(self, directory):
        """directory holds sheet.jsonl and blind-*.jsonl from eval run."""
        self.directory = Path(directory)
        self.lock = threading.Lock()

    def _find(self, name):
        """Returns the sheet item for name, or raises KeyError."""
        for item in sheets(self.directory):
            if item["name"] == name:
                return item
        raise KeyError(name)

    def listing(self):
        """Returns name, kind, rated, and total for every sheet."""
        with self.lock:
            return [{key: item[key]
                     for key in ("name", "kind", "rated", "total")}
                    for item in sheets(self.directory)]

    def cases(self, name):
        """Returns every row of one sheet."""
        with self.lock:
            return list(rows(self._find(name)["path"]))

    def rate(self, name, payload):
        """Saves one case's rating and returns the updated row.

        Continuity payloads carry id and replies [{label, satisfies}] with
        satisfies true, false, or null. Blind payloads carry id and winner
        (A, B, tie, or empty). Raises KeyError for an unknown sheet or case
        and ValueError for a bad rating.
        """
        with self.lock:
            item = self._find(name)
            data = list(rows(item["path"]))
            row = next((r for r in data if r["id"] == payload.get("id")),
                       None)
            if row is None:
                raise KeyError(payload.get("id"))
            if item["kind"] == "continuity":
                apply_replies(row, payload.get("replies"))
            else:
                apply_winner(row, payload.get("winner"))
            write_atomic(item["path"], "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in data))
            return row


def apply_replies(row, marks):
    """Sets satisfies on a continuity row from [{label, satisfies}]."""
    if not isinstance(marks, list):
        raise ValueError("Expected replies.")
    by_label = {reply["label"]: reply for reply in row["replies"]}
    chosen = {}
    for mark in marks:
        label = mark.get("label") if isinstance(mark, dict) else None
        if label not in by_label:
            raise ValueError(f"Unknown reply label: {label}")
        value = mark.get("satisfies")
        if value is not None and type(value) is not bool:
            raise ValueError("satisfies must be true, false, or null.")
        chosen[label] = value
    for label, value in chosen.items():
        by_label[label]["satisfies"] = value


def apply_winner(row, winner):
    """Sets winner on a blind row."""
    if winner not in WINNERS:
        raise ValueError("winner must be A, B, tie, or empty.")
    row["winner"] = winner


def make_server(directory, port):
    """Returns a server bound to 127.0.0.1. port 0 picks a free port."""
    store = Store(directory)

    class Handler(BaseHTTPRequestHandler):
        """Serves the page and the rating JSON routes."""

        def do_GET(self):
            """Handles the page, the sheet list, and one sheet."""
            self._route("GET")

        def do_POST(self):
            """Handles saving one rating."""
            self._route("POST")

        def _route(self, method):
            """Dispatches one request, or returns a JSON error."""
            path = unquote(urlparse(self.path).path)
            prefix = "/api/sheets/"
            try:
                if method == "GET" and path in ("/", "/index.html"):
                    page = assemble_page(HTML, PAGE_STYLE, PAGE_SCRIPTS)
                    self._send(200, "text/html; charset=utf-8", page)
                elif method == "GET" and path == "/api/sheets":
                    self._json(200, store.listing())
                elif path.startswith(prefix) and method == "GET":
                    self._json(200, store.cases(path[len(prefix):]))
                elif path.startswith(prefix) and method == "POST":
                    row = store.rate(path[len(prefix):], self._read_json())
                    self._json(200, {"ok": True, "row": row})
                else:
                    self._json(404, {"error": "Not found."})
            except KeyError:
                self._json(404, {"error": "Unknown sheet or case."})
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
            """Drops the default access log."""
            return

    return ThreadingHTTPServer((HOST, port), Handler)


def main():
    """Serves the rating page until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    missing = [p.name for p in page_paths(HTML, PAGE_STYLE, PAGE_SCRIPTS)
               if not p.is_file()]
    if missing:
        raise SystemExit("Missing " + ", ".join(missing))
    if not sheets(args.dir):
        raise SystemExit(f"No rating sheets in {args.dir}. Run eval run.")
    server = make_server(args.dir, args.port)
    print(f"Rate UI ({args.dir}) at "
          f"http://{HOST}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
