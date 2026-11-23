"""Checks that the curated export rebuilds the same datasets as full PIPPA.

Run: python -m unittest discover tests
"""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from storythread import publish  # noqa: E402
from storythread.data import rows  # noqa: E402
from test_fixture import FIXTURES, run  # noqa: E402

FILES = ("recent/train.jsonl", "recent/valid.jsonl", "recent/manifest.jsonl",
         "earlier/train.jsonl", "earlier/valid.jsonl", "earlier/manifest.jsonl",
         "summary.json")


class PublishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, source, review, name):
        out = self.root / name
        run("data", "prepare", source, review, FIXTURES / "authored.jsonl",
            out, "--valid-share", 0.25, "--seed", 3)
        return out

    def test_subset_rebuilds_identical_datasets(self):
        curated = self.root / "curated"
        count = publish.export(
            FIXTURES / "pippa-sample.jsonl", FIXTURES / "review.jsonl", curated)
        approved = [r for r in rows(FIXTURES / "review.jsonl")
                    if r["approved"] is True]
        self.assertEqual(count, len(approved))
        full = self.prepare(
            FIXTURES / "pippa-sample.jsonl", FIXTURES / "review.jsonl", "full")
        small = self.prepare(
            curated / "pippa-subset.jsonl", curated / "review.jsonl", "small")
        for name in FILES:
            self.assertEqual((full / name).read_bytes(),
                             (small / name).read_bytes(), name)

    def test_subset_keeps_only_read_fields(self):
        curated = self.root / "curated"
        publish.export(
            FIXTURES / "pippa-sample.jsonl", FIXTURES / "review.jsonl", curated)
        for row in rows(curated / "pippa-subset.jsonl"):
            self.assertEqual(set(row), set(publish.KEPT))

    def test_missing_scene_is_refused(self):
        empty = self.root / "empty.jsonl"
        empty.write_text("")
        with self.assertRaises(ValueError):
            publish.export(empty, FIXTURES / "review.jsonl", self.root / "out")


if __name__ == "__main__":
    unittest.main()
