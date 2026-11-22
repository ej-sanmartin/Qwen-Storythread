"""Local page for rating blinded eval replies.

Reads the sheets that `eval run` writes (continuity sheet and engagement
pairs) and saves each rating straight back to its file. The key files stay
on disk: the page never learns which model wrote which reply.

Run: python -m storythread.rate_ui [--dir work/eval]
"""
import argparse
import json
import threading
from pathlib import Path

from .data import rows
from .eval_flow import DEFAULT_DIR, sheets
from .http_ui import (
    HOST, JsonHandler, assemble_page, bind, page_paths, write_atomic)

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

    class Handler(JsonHandler):
        """Serves the page and the rating JSON routes."""

        key_error = "Unknown sheet or case."

        def route(self, method, path):
            """Answers the page, the sheet list, and one rating."""
            prefix = "/api/sheets/"
            if method == "GET" and path in ("/", "/index.html"):
                page = assemble_page(HTML, PAGE_STYLE, PAGE_SCRIPTS)
                self._send(200, "text/html; charset=utf-8", page)
                return True
            if method == "GET" and path == "/api/sheets":
                self._json(200, store.listing())
                return True
            if path.startswith(prefix) and method == "GET":
                self._json(200, store.cases(path[len(prefix):]))
                return True
            if path.startswith(prefix) and method == "POST":
                row = store.rate(path[len(prefix):], self._read_json())
                self._json(200, {"ok": True, "row": row})
                return True
            return False

    return bind(Handler, port)


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
