"""Checks the local review page against the synthetic fixture.

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

from storythread import review_ui  # noqa: E402
from storythread.data import rows  # noqa: E402
from test_fixture import FIXTURES, run  # noqa: E402

PENDING = "8f071e62285ca8d51b34be3e21a76b70ddb89816d5a4d2b00ce4371e834301bf"
WICK = "f308db8b980ef4a464595aba54e325542664267f7f367a665f96476b0852a518"
CARD = "Reviewed Brannoc card."
WICK_CARD = (
    "Wick, a lamplighter. Speak plainly and leave the player's choices open.")


def suggestion(model, total, start, end, reason):
    """Returns one scored suggestion row for the pending fixture scene."""
    return {
        "scene_id": PENDING, "name": "Not the index name", "judge_model": model,
        "status": "proposed", "start": start, "end": end, "total": total,
        "keep_reason": reason, "concern": "Short.", "flags": ["empty_card"],
    }


class ReviewUiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.review = self.root / "review.jsonl"
        self.index = self.root / "index.jsonl"
        self.suggestions = self.root / "suggestions.jsonl"
        run("data", "index", FIXTURES / "pippa-sample.jsonl", self.index)
        shutil.copyfile(FIXTURES / "review.jsonl", self.review)
        rows = [
            suggestion("qwen", 6, 0, 7, "First judge."),
            suggestion("claude", 8, 2, 7, "Later judge."),
            {"scene_id": PENDING, "judge_model": "claude", "status": "skipped",
             "skip_reason": "Trailing skip without a score.", "flags": []},
        ]
        text = "".join(json.dumps(row) + "\n" for row in rows)
        self.suggestions.write_text(text, encoding="utf-8")
        self.server = review_ui.make_server(
            self.review, self.index, self.suggestions, 0)
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
        """Polls until the server accepts a queue request."""
        deadline = time.time() + 2
        while time.time() < deadline:
            try:
                self.request("GET", "/api/queue")
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("review UI did not start")

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
        if path == "/" or path.startswith("/scene/"):
            return status, raw
        return status, json.loads(raw.decode())

    def test_page_lists_the_three_actions(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Approve and next", body)
        self.assertIn(b"Skip and next", body)
        self.assertIn(b"Return to pending", body)
        scene_page = self.request("GET", "/scene/" + PENDING)
        self.assertEqual(scene_page[0], 200)
        self.assertIn(b"Approve and next", scene_page[1])

    def test_decisions_persist_and_prepare_still_reads_approved(self):
        status, queue = self.request("GET", "/api/queue")
        self.assertEqual(status, 200)
        by_id = {row["scene_id"]: row for row in queue}
        self.assertEqual(by_id[PENDING]["status"], "pending")
        self.assertEqual(by_id[PENDING]["name"], "Brannoc")
        self.assertEqual(by_id[PENDING]["score"], 8)
        self.assertEqual(by_id[PENDING]["suggested_start"], 2)
        self.assertEqual(by_id[PENDING]["suggested_end"], 7)
        self.assertEqual(by_id[PENDING]["suggestion_count"], 3)
        self.assertEqual(by_id[WICK]["status"], "approved")
        self.assertIsNone(by_id[WICK]["score"])
        self.assertEqual(
            sum(row["status"] == "pending" for row in queue), 1)
        self.assertEqual(
            sum(row["status"] == "approved" for row in queue), 13)

        status, scene = self.request("GET", f"/api/scene/{PENDING}")
        self.assertEqual(status, 200)
        self.assertEqual(len(scene["turns"]), 7)
        self.assertEqual(scene["suggestions"][1]["keep_reason"], "Later judge.")
        self.assertFalse(scene["approved"])

        original = self.review.read_bytes()
        bad = {"system": "nope", "start": 0, "end": 6,
               "review_status": "approved"}
        status, body = self.request("POST", f"/api/scene/{PENDING}", bad)
        self.assertEqual(status, 400)
        self.assertIn("assistant", body["error"])
        self.assertEqual(self.review.read_bytes(), original)
        bad["end"] = 8
        status, body = self.request("POST", f"/api/scene/{PENDING}", bad)
        self.assertEqual(status, 400)
        self.assertIn("outside", body["error"])
        self.assertEqual(self.review.read_bytes(), original)

        before = self.review.read_text(encoding="utf-8").splitlines()
        saved = {"system": CARD, "start": 0, "end": 7,
                 "review_status": "approved"}
        status, body = self.request("POST", f"/api/scene/{PENDING}", saved)
        self.assertEqual(status, 200, body)
        after = self.review.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(before), len(after))
        changed = [i for i, pair in enumerate(zip(before, after))
                   if pair[0] != pair[1]]
        self.assertEqual(changed, [len(before) - 1])

        status, scene = self.request("GET", f"/api/scene/{PENDING}")
        self.assertEqual(scene["status"], "approved")
        self.assertTrue(scene["approved"])
        self.assertEqual(scene["system"], CARD)
        self.assertEqual(scene["start"], 0)
        self.assertEqual(scene["end"], 7)
        fresh = review_ui.Store(self.review, self.index, self.suggestions)
        self.assertEqual(fresh.scene_view(PENDING)["status"], "approved")
        self.assertTrue(fresh.scene_view(PENDING)["approved"])
        row = json.loads(after[-1])
        self.assertEqual(row["review_status"], "approved")
        self.assertTrue(row["approved"])
        self.assertEqual(row["group"], "bot-13")
        self.assertEqual(row["tags"], ["fantasy"])

        status, wick = self.request("GET", f"/api/scene/{WICK}")
        self.assertEqual(wick["status"], "approved")
        self.assertNotIn("review_status", json.loads(after[0]))

        skipped = {"system": CARD, "start": 99, "end": 100,
                   "review_status": "skipped"}
        status, body = self.request("POST", f"/api/scene/{PENDING}", skipped)
        self.assertEqual(status, 200, body)
        status, scene = self.request("GET", f"/api/scene/{PENDING}")
        self.assertEqual(scene["status"], "skipped")
        self.assertFalse(scene["approved"])
        self.assertEqual(scene["start"], 99)

        pending = {"system": CARD, "start": 0, "end": 7,
                   "review_status": "pending"}
        status, body = self.request("POST", f"/api/scene/{PENDING}", pending)
        self.assertEqual(status, 200, body)
        status, scene = self.request("GET", f"/api/scene/{PENDING}")
        self.assertEqual(scene["status"], "pending")
        self.assertFalse(scene["approved"])

        status, body = self.request("POST", f"/api/scene/{PENDING}", saved)
        self.assertEqual(status, 200, body)
        wick_skip = {
            "system": wick["system"], "start": wick["start"],
            "end": wick["end"],
            "review_status": "skipped",
        }
        status, body = self.request("POST", f"/api/scene/{WICK}", wick_skip)
        self.assertEqual(status, 200, body)
        untouched = self.review.read_text(encoding="utf-8").splitlines()
        self.assertEqual(untouched[1], before[1])

        out = self.root / "datasets"
        run("data", "prepare", FIXTURES / "pippa-sample.jsonl", self.review,
            FIXTURES / "authored.jsonl", out, "--valid-share", "0.25",
            "--authored-share", "0.01", "--seed", "3")
        manifest = [row for row in rows(out / "recent" / "manifest.jsonl")
                    if row["source"] == "PygmalionAI/PIPPA"]
        ids = {row["scene_id"] for row in manifest}
        self.assertIn(PENDING, ids)
        self.assertNotIn(WICK, ids)
        self.assertEqual(len(ids), 13)
        exported = []
        for split in ("train", "valid"):
            path = out / "recent" / f"{split}.jsonl"
            exported += list(rows(path))
        cards = [row["messages"][0]["content"] for row in exported]
        self.assertIn(CARD, cards)
        self.assertNotIn(WICK_CARD, cards)
        kept = next(row for row in exported
                    if row["messages"][0]["content"] == CARD)
        self.assertEqual(kept["messages"][-2]["role"], "user")
        self.assertEqual(kept["messages"][-1]["role"], "assistant")

        missing = self.request("GET", "/api/scene/missing")
        self.assertEqual(missing[0], 404)
