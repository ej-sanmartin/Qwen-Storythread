"""Checks `eval run`, `eval report`, and the rating page on the fixture.

Model generation is replaced by synthetic outputs.
Run: python -m unittest discover tests
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from storythread import eval as evaluation  # noqa: E402
from storythread import eval_flow, rate_ui  # noqa: E402
from storythread.data import rows, write_rows  # noqa: E402
from test_fixture import FIXTURES, run  # noqa: E402

CASES = FIXTURES / "eval-test.jsonl"


def fake_generate(args, label, adapter, target):
    """Stands in for eval_flow.generate_condition without a model."""
    sha = evaluation.file_sha(args.cases)
    write_rows(target, ({
        "id": cid, "output": f"{label} reply to {cid}", "condition": label,
        "prompt_sha": cid, "max_new_tokens": args.max_tokens,
        "cases_sha": sha} for cid in evaluation.load_cases(args.cases)))


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "eval"
        self.original = eval_flow.generate_condition
        eval_flow.generate_condition = fake_generate

    def tearDown(self):
        eval_flow.generate_condition = self.original
        self.tmp.cleanup()

    def args(self, *adapters):
        return argparse.Namespace(
            cases=CASES.as_posix(), adapter=list(adapters),
            dir=self.out.as_posix(), model="m", max_tokens=160, seed=42)

    def test_run_builds_every_sheet_once(self):
        eval_flow.run(self.args("recent=r", "earlier=e"))
        names = sorted(p.name for p in self.out.glob("*.jsonl"))
        self.assertEqual(names, [
            "base.jsonl", "blind-base-earlier-key.jsonl",
            "blind-base-earlier.jsonl", "blind-base-recent-key.jsonl",
            "blind-base-recent.jsonl", "blind-recent-earlier-key.jsonl",
            "blind-recent-earlier.jsonl", "earlier.jsonl", "key.jsonl",
            "recent.jsonl", "sheet.jsonl"])
        self.assertEqual(len(list(rows(self.out / "sheet.jsonl"))), 6)

    def test_rerun_keeps_ratings(self):
        eval_flow.run(self.args("recent=r"))
        sheet = self.out / "sheet.jsonl"
        data = list(rows(sheet))
        data[0]["replies"][0]["satisfies"] = True
        sheet.unlink()
        write_rows(sheet, data)
        eval_flow.run(self.args("recent=r"))
        self.assertIs(list(rows(sheet))[0]["replies"][0]["satisfies"], True)

    def test_new_adapter_refuses_to_replace_rated_sheet(self):
        eval_flow.run(self.args("recent=r"))
        with self.assertRaises(SystemExit) as caught:
            eval_flow.run(self.args("recent=r", "earlier=e"))
        self.assertIn("Delete", str(caught.exception))

    def test_stale_outputs_are_rejected(self):
        eval_flow.run(self.args())
        base = self.out / "base.jsonl"
        data = [dict(row, cases_sha="old") for row in rows(base)]
        base.unlink()
        write_rows(base, data)
        with self.assertRaises(SystemExit):
            eval_flow.run(self.args())

    def test_bad_adapter_labels_are_rejected(self):
        for bad in ("nolabel", "base=x", "a/b=x", "dup=x"):
            args = ["dup=y", bad] if bad == "dup=x" else [bad]
            with self.assertRaises(SystemExit):
                eval_flow.parse_adapters(args)

    def test_report_tracks_progress_then_scores(self):
        eval_flow.run(self.args("recent=r"))
        report = argparse.Namespace(
            cases=CASES.as_posix(), dir=self.out.as_posix(), edges="4,12")
        eval_flow.report(report)
        partial = json.loads((self.out / "report.json").read_text())
        self.assertEqual(partial["sheet"]["status"], "0 of 12 rated")
        self.assertEqual(partial["blind-base-recent"]["status"], "0 of 6 rated")

        sheet, key = self.out / "sheet.jsonl", self.out / "key.jsonl"
        labels = {k["id"]: k["labels"] for k in rows(key)}
        data = list(rows(sheet))
        for row in data:
            for reply in row["replies"]:
                reply["satisfies"] = (
                    labels[row["id"]][reply["label"]] == "recent")
        sheet.unlink()
        write_rows(sheet, data)
        eval_flow.report(report)
        done = json.loads((self.out / "report.json").read_text())
        self.assertEqual(done["sheet"]["recent"]["both_versions"],
                         {"count": 3, "total": 3, "percent": 100.0})
        self.assertEqual(done["sheet"]["base"]["both_versions"]["count"], 0)

    def test_cli_registers_run_and_report(self):
        result = run("eval", "run", "--help")
        self.assertIn("--adapter", result.stdout)
        self.assertIn("--dir", run("eval", "report", "--help").stdout)


class RateUiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        original = eval_flow.generate_condition
        eval_flow.generate_condition = fake_generate
        try:
            eval_flow.run(argparse.Namespace(
                cases=CASES.as_posix(), adapter=["recent=r"],
                dir=self.out.as_posix(), model="m", max_tokens=160, seed=42))
        finally:
            eval_flow.generate_condition = original
        self.server = rate_ui.make_server(self.out, 0)
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
                self.request("GET", "/api/sheets")
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("rate UI did not start")

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
        return status, raw if path == "/" else json.loads(raw.decode())

    def test_page_and_listing(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Storythread Rate", page)
        _, listing = self.request("GET", "/api/sheets")
        self.assertEqual(
            [(s["name"], s["kind"], s["rated"], s["total"]) for s in listing],
            [("sheet", "continuity", 0, 12),
             ("blind-base-recent", "blind", 0, 6)])

    def test_cases_hide_which_model_wrote_what(self):
        _, data = self.request("GET", "/api/sheets/sheet")
        text = json.dumps(data)
        for secret in ("base", "recent"):
            self.assertNotIn(f'"condition": "{secret}"', text)
        self.assertEqual(sorted(data[0]), [
            "evidence", "evidence_turn", "expected", "id", "messages",
            "replies", "scenario_id"])

    def test_rating_continuity_persists(self):
        _, data = self.request("GET", "/api/sheets/sheet")
        row = data[0]
        marks = [{"label": r["label"], "satisfies": i % 2 == 0}
                 for i, r in enumerate(row["replies"])]
        status, _ = self.request(
            "POST", "/api/sheets/sheet", {"id": row["id"], "replies": marks})
        self.assertEqual(status, 200)
        saved = list(rows(self.out / "sheet.jsonl"))[0]
        self.assertEqual([r["satisfies"] for r in saved["replies"]],
                         [True, False])
        self.assertEqual(saved["replies"][0]["text"], row["replies"][0]["text"])
        _, listing = self.request("GET", "/api/sheets")
        self.assertEqual(listing[0]["rated"], 2)
        marks[0]["satisfies"] = None
        self.request(
            "POST", "/api/sheets/sheet", {"id": row["id"], "replies": marks})
        _, listing = self.request("GET", "/api/sheets")
        self.assertEqual(listing[0]["rated"], 1)

    def test_rating_blind_winner(self):
        _, data = self.request("GET", "/api/sheets/blind-base-recent")
        status, _ = self.request("POST", "/api/sheets/blind-base-recent",
                                 {"id": data[0]["id"], "winner": "tie"})
        self.assertEqual(status, 200)
        self.assertEqual(
            list(rows(self.out / "blind-base-recent.jsonl"))[0]["winner"],
            "tie")
        _, listing = self.request("GET", "/api/sheets")
        self.assertEqual(listing[1]["rated"], 1)

    def test_bad_ratings_do_not_write(self):
        before = (self.out / "sheet.jsonl").read_bytes()
        _, data = self.request("GET", "/api/sheets/sheet")
        row = data[0]
        cases = [
            ("/api/sheets/sheet", {"id": row["id"], "replies": [
                {"label": "R9", "satisfies": True}]}, 400),
            ("/api/sheets/sheet", {"id": row["id"], "replies": [
                {"label": "R1", "satisfies": "yes"}]}, 400),
            ("/api/sheets/sheet", {"id": "nope", "replies": []}, 404),
            ("/api/sheets/blind-base-recent",
             {"id": data[0]["id"], "winner": "C"}, 400),
            ("/api/sheets/missing", {"id": "x"}, 404),
            ("/api/sheets/..%2Fkey", {"id": "x"}, 404),
        ]
        for path, payload, expected in cases:
            status, _ = self.request("POST", path, payload)
            self.assertEqual(status, expected, path)
        self.assertEqual((self.out / "sheet.jsonl").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
