"""Local reviewer for work/review.jsonl.

Serves one HTML page on 127.0.0.1. Decisions are written back to the review
file; data.py prepare still selects rows with approved true.

Run: python -m storythread.review_ui
"""
import argparse
from pathlib import Path

from .http_ui import (
    HOST, JsonHandler, assemble_page, bind, page_paths)
from .review_store import Store

PORT = 8765
HTML = Path(__file__).parent / "web" / "review" / "review_ui.html"
PAGE_STYLE = "review_ui.css"
PAGE_SCRIPTS = ("review_ui_queue.js", "review_ui_scene.js")


def is_page(path):
    """True for the queue and for /scene/<scene_id>."""
    if path in ("/", "/index.html"):
        return True
    return path.startswith("/scene/") and path != "/scene/"


def make_server(review, index, suggestions, port):
    """Returns a server bound to 127.0.0.1. port 0 picks a free port."""
    store = Store(review, index, suggestions)

    class Handler(JsonHandler):
        """Serves the page and the review JSON routes."""

        key_error = "Unknown scene."

        def route(self, method, path):
            """Answers the queue, one scene, and the HTML page."""
            if method == "GET" and is_page(path):
                page = assemble_page(HTML, PAGE_STYLE, PAGE_SCRIPTS)
                self._send(200, "text/html; charset=utf-8", page)
                return True
            if method == "GET" and path == "/api/queue":
                self._json(200, store.queue())
                return True
            prefix = "/api/scene/"
            if path.startswith(prefix) and len(path) > len(prefix):
                scene_id = path[len(prefix):]
                if method == "GET":
                    self._json(200, store.scene_view(scene_id))
                    return True
                if method == "POST":
                    payload = self._read_json()
                    store.decide(
                        scene_id, payload.get("system"),
                        payload.get("start"), payload.get("end"),
                        payload.get("review_status"))
                    self._json(200, {"ok": True})
                    return True
            return False

    return bind(Handler, port)


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
