"""Shared pieces of the local pages: files, page assembly, JSON handler."""
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HOST = "127.0.0.1"
MAX_BODY = 2_000_000


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


def bind(handler, port):
    """Returns a server bound to 127.0.0.1. port 0 picks a free port."""
    return ThreadingHTTPServer((HOST, port), handler)


class JsonHandler(BaseHTTPRequestHandler):
    """GET, POST, and DELETE dispatcher with JSON errors.

    Subclasses implement route(method, path) and return True when the
    request was answered. key_error is the 404 text for a KeyError.
    """

    key_error = "Not found."

    def do_GET(self):
        """Dispatches a GET request."""
        self.dispatch("GET")

    def do_POST(self):
        """Dispatches a POST request."""
        self.dispatch("POST")

    def do_DELETE(self):
        """Dispatches a DELETE request."""
        self.dispatch("DELETE")

    def dispatch(self, method):
        """Runs handle, or returns a JSON error."""
        path = unquote(urlparse(self.path).path)
        try:
            if not self.route(method, path):
                self._json(404, {"error": "Not found."})
        except KeyError:
            self._json(404, {"error": self.key_error})
        except ValueError as error:
            self._json(400, {"error": str(error)})

    def route(self, method, path):
        """Returns True when this handler answered the request."""
        return False

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
