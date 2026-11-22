"""Checks the authoring page in eval mode (author_ui --kind eval).

Run: python -m unittest discover tests
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from storythread import author_ui  # noqa: E402
from storythread.author_eval import sibling  # noqa: E402
from storythread.data import rows  # noqa: E402
from storythread.scenarios import parse_scenarios  # noqa: E402
from test_fixture import FIXTURES  # noqa: E402


def version(fruit, quote):
    """Returns one eval version whose second turn mentions fruit."""
    return {
        "messages": [
            {"role": "system", "content": "Orla, a market seller."},
            {"role": "assistant", "content": "Orla waves from her stall."},
            {"role": "user", "content": f"I am allergic to {fruit}."},
            {"role": "assistant", "content": "She nods slowly."},
            {"role": "user", "content": "What should I buy?"},
        ],
        "expected": f"Avoids {fruit}.", "evidence_turn": 1, "evidence": quote,
    }


NEW = {
    "scenario_id": "dev-market", "group": "orla",
    "versions": {"a": version("pears", "pears"),
                 "b": version("plums", "plums")},
}


class EvalModeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dev = self.root / "dev.jsonl"
        self.test = self.root / "test.jsonl"
        shutil.copyfile(FIXTURES / "eval-dev.jsonl", self.dev)
        shutil.copyfile(FIXTURES / "eval-test.jsonl", self.test)
        self.server = author_ui.make_server(
            self.dev, 0, kind="eval", other=self.test)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.wait_until_up()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def wait_until_up(self):
        deadline = time.time() + 2
        while time.time() < deadline:
            try:
                self.request("GET", "/api/info")
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("author UI did not start")

    def request(self, method, path, payload=None):
        """Returns (status, parsed JSON or page bytes)."""
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, headers=headers,
            method=method)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        page = path in ("/", "/new") or path.startswith("/scene/")
        return status, raw if page else json.loads(raw.decode())

    def test_page_serves_the_eval_editor(self):
        for path in ("/", "/new", "/scene/dev-lockbox"):
            status, body = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertIn(b"Use as evidence turn", body)
            self.assertIn(b"Final turn, the same in both versions", body)
            self.assertNotIn(b"Earlier evidence", body)
        _, info = self.request("GET", "/api/info")
        self.assertEqual(info, {"kind": "eval", "file": str(self.dev)})

    def test_lists_scenarios_with_turn_counts(self):
        _, listed = self.request("GET", "/api/scenes")
        self.assertEqual(
            [(r["scenario_id"], r["turns"]) for r in listed],
            [("dev-lockbox", 5), ("dev-bridge", 11)])

    def test_token_check_is_scenes_only(self):
        status, _ = self.request("POST", "/api/tokens", dict(NEW))
        self.assertEqual(status, 404)

    def test_save_appends_a_valid_scenario(self):
        before = self.dev.read_text(encoding="utf-8").splitlines()
        status, body = self.request("POST", "/api/scenes", NEW)
        self.assertEqual(status, 200, body)
        after = self.dev.read_text(encoding="utf-8").splitlines()
        self.assertEqual(after[:-1], before)
        parsed = parse_scenarios(list(rows(self.dev)))
        self.assertEqual(parsed[-1]["distance"], 3)

    def test_save_replaces_one_row_and_keeps_the_rest(self):
        before = self.dev.read_text(encoding="utf-8").splitlines()
        edited = json.loads(before[0])
        edited["versions"]["a"]["expected"] = "  Says the code.  "
        status, body = self.request(
            "POST", "/api/scenes/dev-lockbox", edited)
        self.assertEqual(status, 200, body)
        after = self.dev.read_text(encoding="utf-8").splitlines()
        self.assertEqual(after[1:], before[1:])
        self.assertEqual(json.loads(after[0])["versions"]["a"]["expected"],
                         "Says the code.")

    def test_invalid_scenarios_are_rejected_without_writing(self):
        original = self.dev.read_bytes()
        same = copy.deepcopy(NEW)
        same["versions"]["b"] = copy.deepcopy(same["versions"]["a"])
        missing = copy.deepcopy(NEW)
        missing["versions"]["a"]["evidence"] = "apricots"
        tail = copy.deepcopy(NEW)
        tail["versions"]["b"]["messages"][-1]["content"] = "Anything else?"
        dup = copy.deepcopy(NEW)
        dup["scenario_id"] = "dev-lockbox"
        for payload, word in ((same, "differ"), (missing, "not found"),
                              (tail, "final user prompt"),
                              (dup, "Duplicate"), ({"versions": 1}, "a and b")):
            status, body = self.request("POST", "/api/scenes", payload)
            self.assertEqual(status, 400, payload)
            self.assertIn(word, body["error"])
        self.assertEqual(self.dev.read_bytes(), original)

    def test_dev_cannot_reuse_a_test_group_or_card(self):
        original = self.dev.read_bytes()
        clash = copy.deepcopy(NEW)
        clash["group"] = "Perrin"
        status, body = self.request("POST", "/api/scenes", clash)
        self.assertEqual(status, 400)
        self.assertIn("group", body["error"])
        self.assertEqual(self.dev.read_bytes(), original)

    def test_delete_removes_one_scenario(self):
        status, _ = self.request("DELETE", "/api/scenes/dev-bridge")
        self.assertEqual(status, 200)
        self.assertEqual(
            [r["scenario_id"] for r in rows(self.dev)], ["dev-lockbox"])

    def test_sibling_swaps_dev_and_test(self):
        self.assertEqual(sibling("eval/dev.jsonl"), Path("eval/test.jsonl"))
        self.assertEqual(sibling("eval/test.jsonl"), Path("eval/dev.jsonl"))
        self.assertIsNone(sibling("eval/other.jsonl"))


if __name__ == "__main__":
    unittest.main()
