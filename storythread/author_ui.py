"""Local editor for authored/scenes.jsonl or eval/dev.jsonl.

Serves one HTML page on 127.0.0.1. Saves scene pairs in the authored format,
or with --kind eval, two-version eval scenarios. Token checks use the same
measurement as runtime.preflight.

Run: python -m storythread.author_ui
     python -m storythread.author_ui --kind eval
"""
import argparse
from pathlib import Path

from .author_eval import sibling
from .author_store import Store, require_quotes, scene_from
from .export import authored_example
from .http_ui import HOST, JsonHandler, assemble_page, bind, page_paths
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


def token_report(store, payload, cap):
    """Measures both endings the way runtime.preflight measures a row.

    cap is the training token limit. Returns max_tokens plus a recent and
    earlier report. Does not write the scene file.
    """
    if type(cap) is not int or cap < 1:
        raise ValueError("max_tokens must be a positive integer.")
    scene = scene_from(payload)
    require_quotes([scene])
    parsed = parse_authored([scene])[0]
    tokenizer = store.tokenizer()
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
            "evidence_distance_tokens": info.get("evidence_distance_tokens"),
            "distance_turns": manifest["evidence_distance_turns"],
        }
    return report


def is_page(path):
    """True for the editor shell, including /new and /scene/."""
    if path in ("/", "/index.html", "/new"):
        return True
    return path.startswith("/scene/") and path != "/scene/"


def make_server(scenes, port, tokenizer=None, model=DEFAULT_MODEL,
                kind="scenes", other=None, eval_dir="work/eval"):
    """Returns a server bound to 127.0.0.1. port 0 picks a free port.

    kind, other, and eval_dir are described on Store.
    """
    store = Store(scenes, tokenizer=tokenizer, model=model, kind=kind,
                  other=other, eval_dir=eval_dir)
    scripts = SCRIPTS[kind]

    class Handler(JsonHandler):
        """Serves the page and the authoring JSON routes."""

        key_error = "Unknown scene."

        def route(self, method, path):
            """Answers the page, the scene list, and one scene."""
            if method == "GET" and is_page(path):
                page = assemble_page(HTML, PAGE_STYLE, scripts)
                self._send(200, "text/html; charset=utf-8", page)
                return True
            if method == "GET" and path == "/api/info":
                self._json(200, {"kind": kind, "file": str(store.path)})
                return True
            if method == "GET" and path == "/api/scenes":
                self._json(200, store.list_scenes())
                return True
            if method == "POST" and path == "/api/scenes":
                scene = store.save(None, self._read_json())
                self._json(200, {"ok": True, "scene": scene})
                return True
            if method == "POST" and path == "/api/tokens" and kind == "scenes":
                payload = self._read_json()
                report = token_report(
                    store, payload, payload.get("max_tokens", 512))
                self._json(200, report)
                return True
            prefix = "/api/scenes/"
            if path.startswith(prefix) and len(path) > len(prefix):
                scenario_id = path[len(prefix):]
                if method == "GET":
                    self._json(200, store.get_scene(scenario_id))
                    return True
                if method == "POST":
                    scene = store.save(scenario_id, self._read_json())
                    self._json(200, {"ok": True, "scene": scene})
                    return True
                if method == "DELETE":
                    store.delete(scenario_id)
                    self._json(200, {"ok": True})
                    return True
            return False

    return bind(Handler, port)


def main():
    """Serves the authoring page until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(SCRIPTS), default="scenes")
    parser.add_argument(
        "--scenes", help="file to edit (default depends on --kind)")
    parser.add_argument(
        "--other", help="sibling eval file; default swaps dev and test")
    parser.add_argument(
        "--eval-dir", default="work/eval",
        help="generation outputs that lock an eval file")
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
                         kind=args.kind, other=other, eval_dir=args.eval_dir)
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
