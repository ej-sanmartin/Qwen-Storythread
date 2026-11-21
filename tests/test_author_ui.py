"""Checks the local authoring page against the synthetic fixture.

Run: python -m unittest discover tests
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from storythread import author_ui  # noqa: E402
from storythread.data import rows  # noqa: E402
from storythread.scenarios import parse_authored  # noqa: E402
from test_fixture import FIXTURES, FakeTokenizer, run  # noqa: E402

NEW = {
    "scenario_id": "editor-well",
    "group": "nima",
    "system": "Nima, a well keeper who answers plainly.",
    "history": [
        {"role": "assistant", "content": "Nima rests a hand on the stone rim."},
        {"role": "user",
         "content": "The rope snapped when I drew water for the baker."},
        {"role": "assistant",
         "content": "She holds up a spare coil of red rope."},
    ],
    "branches": {
        "recent": {
            "user": "I take the coil.",
            "reply": "Nima nods. 'Tie it twice, then draw.'",
            "evidence_turn": 3,
            "evidence": "coil",
        },
        "earlier": {
            "user": "The baker is still waiting.",
            "reply": "'Tell the baker the rope snapped,' Nima says.",
            "evidence_turn": 1,
            "evidence": "rope snapped",
        },
    },
}


class AuthorUiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scenes = self.root / "scenes.jsonl"
        shutil.copyfile(FIXTURES / "authored.jsonl", self.scenes)
        self.server = author_ui.make_server(
            self.scenes, 0, tokenizer=FakeTokenizer())
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.wait_until_up()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def wait_until_up(self):
        """Polls until the server accepts a list request."""
        deadline = time.time() + 2
        while time.time() < deadline:
            try:
                self.request("GET", "/api/scenes")
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("author UI did not start")

    def request(self, method, path, payload=None):
        """Returns (status, body) for one local request."""
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(
            url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            raw = error.read()
            status = error.code
        if path in ("/", "/new") or path.startswith("/scene/"):
            return status, raw
        return status, json.loads(raw.decode())

    def test_page_has_authoring_controls(self):
        self.assertEqual(author_ui.PORT, 8766)
        for path in ("/", "/new", "/scene/salt-river"):
            status, body = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertIn(b"New scene", body)
            self.assertIn(b"Recent evidence", body)
            self.assertIn(b"Earlier evidence", body)
            self.assertIn(b"Check token counts", body)
            self.assertIn(b"this ending's user turn", body)
            self.assertIn(b"Cited turn was removed", body)

    def test_lists_saved_scenes(self):
        status, listed = self.request("GET", "/api/scenes")
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["scenario_id"] for row in listed],
            ["salt-river", "glass-orchard", "signal-tower", "ink-market"])
        self.assertEqual(listed[0]["turns"], 7)
        missing = self.request("GET", "/api/scenes/missing")
        self.assertEqual(missing[0], 404)

    def test_rejects_invalid_scene_without_writing(self):
        original = self.scenes.read_bytes()
        row = json.loads(original.decode().splitlines()[0])
        row["branches"]["recent"]["evidence"] = "  "
        status, body = self.request("POST", "/api/scenes/salt-river", row)
        self.assertEqual(status, 400)
        self.assertIn("nonempty", body["error"])
        self.assertEqual(self.scenes.read_bytes(), original)

        row["branches"]["recent"]["evidence"] = "not in the turn"
        status, body = self.request("POST", "/api/scenes/salt-river", row)
        self.assertEqual(status, 400)
        self.assertIn("not found", body["error"])
        self.assertEqual(self.scenes.read_bytes(), original)

        row["branches"]["recent"]["evidence"] = "brass bell"
        row["branches"]["earlier"]["evidence_turn"] = 6
        row["branches"]["earlier"]["evidence"] = "brass bell"
        status, body = self.request("POST", "/api/scenes/salt-river", row)
        self.assertEqual(status, 400)
        self.assertIn("farther", body["error"])
        self.assertEqual(self.scenes.read_bytes(), original)

        duplicate = json.loads(json.dumps(NEW))
        duplicate["scenario_id"] = "salt-river"
        status, body = self.request("POST", "/api/scenes", duplicate)
        self.assertEqual(status, 400)
        self.assertIn("Duplicate", body["error"])
        self.assertEqual(self.scenes.read_bytes(), original)

    def test_save_preserves_other_rows(self):
        lines = self.scenes.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["note"] = "keep"
        lines[0] = json.dumps(first, ensure_ascii=False)
        self.scenes.write_text("\n".join(lines) + "\n", encoding="utf-8")
        before = self.scenes.read_text(encoding="utf-8").splitlines()
        edited = json.loads(before[0])
        edited["system"] = "Updated Tamsin card."
        status, body = self.request("POST", "/api/scenes/salt-river", edited)
        self.assertEqual(status, 200, body)
        after = self.scenes.read_text(encoding="utf-8").splitlines()
        self.assertEqual(after[1:], before[1:])
        saved = json.loads(after[0])
        self.assertEqual(saved["system"], "Updated Tamsin card.")
        self.assertEqual(saved["note"], "keep")
        self.assertNotIn("distance_turns", saved["branches"]["recent"])
        parsed = parse_authored(list(rows(self.scenes)))
        self.assertEqual(parsed[0]["system"], "Updated Tamsin card.")

    def test_token_counts_use_preflight_flags(self):
        before = self.scenes.read_bytes()
        status, report = self.request(
            "POST", "/api/tokens", {**NEW, "max_tokens": 100000})
        self.assertEqual(status, 200, report)
        self.assertEqual(report["max_tokens"], 100000)
        self.assertEqual(report["recent"]["distance_turns"], 1)
        self.assertEqual(report["earlier"]["distance_turns"], 3)
        self.assertEqual(report["recent"]["flags"], [])
        self.assertEqual(report["earlier"]["flags"], [])
        self.assertGreater(report["recent"]["evidence_distance_tokens"], 0)
        self.assertGreater(report["earlier"]["evidence_distance_tokens"], 0)
        self.assertGreater(report["recent"]["full"], 0)
        tight = self.request(
            "POST", "/api/tokens", {**NEW, "max_tokens": 1})
        self.assertEqual(tight[0], 200, tight[1])
        self.assertIn("target_truncated", tight[1]["recent"]["flags"])
        self.assertIn("target_truncated", tight[1]["earlier"]["flags"])
        bad = self.request("POST", "/api/tokens", {**NEW, "max_tokens": 0})
        self.assertEqual(bad[0], 400)
        self.assertEqual(self.scenes.read_bytes(), before)

    def test_saved_scene_exports_into_both_arms(self):
        before = self.scenes.read_text(encoding="utf-8").splitlines()
        status, body = self.request("POST", "/api/scenes", NEW)
        self.assertEqual(status, 200, body)
        after = self.scenes.read_text(encoding="utf-8").splitlines()
        self.assertEqual(after[:-1], before)
        saved = json.loads(after[-1])
        self.assertEqual(saved["scenario_id"], "editor-well")
        self.assertEqual(saved["branches"]["recent"]["evidence_turn"], 3)
        self.assertEqual(saved["branches"]["recent"]["evidence"], "coil")
        parsed = parse_authored(list(rows(self.scenes)))
        self.assertIn("editor-well", [item["scenario_id"] for item in parsed])

        out = self.root / "datasets"
        run("data", "prepare", FIXTURES / "pippa-sample.jsonl",
            FIXTURES / "review.jsonl", self.scenes, out,
            "--valid-share", "0.25", "--seed", "3")
        replies = {
            "recent": NEW["branches"]["recent"]["reply"],
            "earlier": NEW["branches"]["earlier"]["reply"],
        }
        distances = {"recent": 1, "earlier": 3}
        turns = {"recent": 3, "earlier": 1}
        for branch, reply in replies.items():
            manifest = [
                row for row in rows(out / branch / "manifest.jsonl")
                if row.get("scenario_id") == "editor-well"]
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["source"], "authored")
            self.assertEqual(manifest[0]["branch"], branch)
            self.assertEqual(manifest[0]["evidence_turn"], turns[branch])
            self.assertEqual(
                manifest[0]["evidence_distance_turns"], distances[branch])
            exported = []
            for split in ("train", "valid"):
                exported += list(rows(out / branch / f"{split}.jsonl"))
            found = [
                row for row in exported
                if row["messages"][-1]["content"] == reply]
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["messages"][-2]["content"],
                             NEW["branches"][branch]["user"])
            self.assertEqual(found[0]["messages"][0]["content"], NEW["system"])
            ids = {
                row["scenario_id"]
                for row in rows(out / branch / "manifest.jsonl")
                if row["source"] == "authored"}
            self.assertIn("salt-river", ids)

    def test_save_creates_a_missing_file(self):
        path = self.root / "fresh" / "scenes.jsonl"
        store = author_ui.Store(path, tokenizer=FakeTokenizer())
        stored = store.save(None, NEW)
        self.assertEqual(stored["scenario_id"], "editor-well")
        self.assertEqual(len(list(rows(path))), 1)
        self.assertEqual(store.list_scenes()[0]["group"], "nima")
