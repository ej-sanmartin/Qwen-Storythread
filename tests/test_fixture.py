"""End-to-end checks on the synthetic fixture. Run: python -m unittest discover tests"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval as evaluation  # noqa: E402
import runtime  # noqa: E402
from data import rows, write_rows  # noqa: E402

FIXTURES = ROOT / "fixtures"
CONDITIONS = ("base", "recent", "earlier")


def run(script, *args, check=True):
    """Runs a CLI script and returns the completed process."""
    result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                            capture_output=True, text=True, cwd=ROOT)
    if check and result.returncode:
        raise AssertionError(result.stderr)
    return result


def write_rows_overwrite(path, data):
    """Replaces a JSONL file (data.write_rows refuses to overwrite)."""
    data = list(data)
    path.unlink()
    write_rows(path, data)


class FakeTokenizer:
    """Whitespace tokenizer with a chat template shaped like Qwen's."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        text = " ".join(f"<s> {m['role']} {m['content']} </s>" for m in messages)
        if add_generation_prompt:
            text += " <s> assistant"
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(t for t in tokens if t not in ("<s>", "</s>"))


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.out = Path(cls.dir.name) / "datasets"
        run("data.py", "prepare", FIXTURES / "pippa-sample.jsonl", FIXTURES / "review.jsonl",
            FIXTURES / "authored.jsonl", cls.out, "--valid-share", 0.25, "--seed", 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def arm(self, branch, split):
        return list(rows(self.out / branch / f"{split}.jsonl"))

    def manifest(self, branch):
        return list(rows(self.out / branch / "manifest.jsonl"))

    def test_arms_share_core_and_counts(self):
        for split in ("train", "valid"):
            recent, earlier = self.arm("recent", split), self.arm("earlier", split)
            self.assertEqual(len(recent), len(earlier))
            meta = [m for m in self.manifest("recent") if m["split"] == split]
            core = [i for i, m in enumerate(meta) if m["source"] != "authored"]
            self.assertEqual([recent[i] for i in core], [earlier[i] for i in core])
            authored = [i for i in range(len(meta)) if i not in core]
            self.assertTrue(all(recent[i] != earlier[i] for i in authored))
            self.assertAlmostEqual(len(authored) / len(meta), 0.25, places=2)

    def test_manifest_records_evidence_and_branch(self):
        for branch in ("recent", "earlier"):
            authored = [m for m in self.manifest(branch) if m["source"] == "authored"]
            self.assertTrue(authored)
            for entry in authored:
                self.assertEqual(entry["branch"], branch)
                self.assertIsInstance(entry["evidence_turn"], int)
        near = {m["scenario_id"]: m["evidence_distance_turns"] for m in self.manifest("recent")
                if m["source"] == "authored"}
        far = {m["scenario_id"]: m["evidence_distance_turns"] for m in self.manifest("earlier")
               if m["source"] == "authored"}
        self.assertTrue(all(far[k] > near[k] for k in near))

    def test_related_cards_stay_in_one_split(self):
        splits = {}
        for entry in self.manifest("recent"):
            splits.setdefault(entry["group"], set()).add(entry["split"])
        self.assertTrue(all(len(s) == 1 for s in splits.values()), splits)
        by_scene = {}
        for entry in self.manifest("recent") + self.manifest("earlier"):
            if entry["source"] == "authored":
                by_scene.setdefault(entry["scenario_id"], set()).add(entry["split"])
        self.assertTrue(all(len(s) == 1 for s in by_scene.values()))

    def test_every_row_ends_with_the_supervised_reply(self):
        for branch in ("recent", "earlier"):
            for split in ("train", "valid"):
                for row in self.arm(branch, split):
                    self.assertEqual(row["messages"][0]["role"], "system")
                    self.assertEqual(row["messages"][-1]["role"], "assistant")

    def test_preflight_reports_lengths_and_flags(self):
        report = runtime.preflight(self.out / "recent", FakeTokenizer(), 100000)
        summary = report["train"]["summary"]
        self.assertIn("authored/recent", summary)
        self.assertGreater(summary["authored/recent"]["evidence_distance_tokens"]["min"], 0)
        self.assertEqual(report["train"]["flagged"], [])
        tight = runtime.preflight(self.out / "earlier", FakeTokenizer(), 30)
        flags = {f for r in tight["train"]["flagged"] for f in r["flags"]}
        self.assertIn("target_truncated", flags)
        self.assertIn("evidence_truncated", flags)

    def test_authored_branches_must_differ_in_distance(self):
        scenes = list(rows(FIXTURES / "authored.jsonl"))
        scenes[0]["branches"]["earlier"]["evidence_turn"] = 6
        scenes[0]["branches"]["earlier"]["evidence"] = "brass bell"
        path = Path(self.dir.name) / "bad.jsonl"
        write_rows(path, scenes)
        result = run("data.py", "prepare", FIXTURES / "pippa-sample.jsonl",
                     FIXTURES / "review.jsonl", path, Path(self.dir.name) / "bad-out",
                     check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("farther", result.stderr)


class EvalTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)
        self.cases = FIXTURES / "eval-test.jsonl"

    def tearDown(self):
        self.dir.cleanup()

    def outputs(self, sha=None):
        sha = sha or evaluation.file_sha(self.cases)
        paths = []
        for condition in CONDITIONS:
            path = self.tmp / f"{condition}.jsonl"
            write_rows(path, ({"id": cid, "output": f"{condition} reply to {cid}",
                               "condition": condition, "prompt_sha": cid,
                               "max_new_tokens": 160, "cases_sha": sha}
                              for cid in evaluation.load_cases(self.cases)))
            paths.append(path)
        return paths

    def test_validate_checks_dev_test_separation(self):
        ok = run("eval.py", "validate", FIXTURES / "eval-dev.jsonl", self.cases)
        self.assertEqual(json.loads(ok.stdout)["test"], 3)
        bad = run("eval.py", "validate", self.cases, self.cases, check=False)
        self.assertNotEqual(bad.returncode, 0)

    def test_continuity_primary_metric_and_distance_breakdown(self):
        paths = self.outputs()
        sheet, key = self.tmp / "sheet.jsonl", self.tmp / "key.jsonl"
        run("eval.py", "sheet", self.cases, sheet, key, "--outputs", *paths)
        labels = {k["id"]: k["labels"] for k in rows(key)}
        self.assertEqual(len({tuple(sorted(v.items())) for v in labels.values()}) > 1, True)
        rated = []
        for row in rows(sheet):
            for reply in row["replies"]:
                condition = labels[row["id"]][reply["label"]]
                reply["satisfies"] = (
                    condition == "earlier"
                    or (condition == "recent" and row["id"] != "test-oven/b"))
            rated.append(row)
        ratings = self.tmp / "rated.jsonl"
        write_rows(ratings, rated)
        out = json.loads(run("eval.py", "continuity", self.cases, ratings, key).stdout)
        self.assertEqual(out["earlier"]["both_versions"],
                         {"count": 3, "total": 3, "percent": 100.0})
        self.assertEqual(out["recent"]["both_versions"]["count"], 2)
        self.assertEqual(out["recent"]["single_versions"]["count"], 5)
        self.assertEqual(out["base"]["both_versions"], {"count": 0, "total": 3, "percent": 0.0})
        near = out["earlier"]["by_evidence_distance_turns"]
        self.assertEqual({k: v["total"] for k, v in near.items()},
                         {"<=4": 1, "5-12": 1, ">12": 1})
        self.assertEqual(out["recent"]["by_evidence_distance_turns"]["<=4"]["count"], 0)

    def test_unrated_replies_are_rejected(self):
        sheet, key = self.tmp / "sheet.jsonl", self.tmp / "key.jsonl"
        run("eval.py", "sheet", self.cases, sheet, key, "--outputs", *self.outputs())
        result = run("eval.py", "continuity", self.cases, sheet, key, check=False)
        self.assertNotEqual(result.returncode, 0)

    def test_mismatched_case_files_are_rejected(self):
        paths = self.outputs()
        write_rows_overwrite(paths[2], [{**r, "cases_sha": "stale"} for r in rows(paths[2])])
        result = run("eval.py", "sheet", self.cases, self.tmp / "s.jsonl", self.tmp / "k.jsonl",
                     "--outputs", *paths, check=False)
        self.assertNotEqual(result.returncode, 0)

    def test_engagement_preference_is_separate(self):
        first, second = self.outputs()[:2]
        public, key = self.tmp / "blind.jsonl", self.tmp / "blind-key.jsonl"
        run("eval.py", "blind", self.cases, first, second, public, key)
        rated = [{**row, "winner": "A"} for row in rows(public)]
        self.assertIn("rather keep chatting", rated[0]["question"])
        self.assertNotIn("expected", rated[0])
        ratings = self.tmp / "blind-rated.jsonl"
        write_rows(ratings, rated)
        out = json.loads(run("eval.py", "score", ratings, key).stdout)
        self.assertEqual(out["cases"], 6)
        self.assertEqual(out["preferred"]["base"] + out["preferred"]["recent"], 6)

    def test_step_metrics_callback_records_time_and_memory(self):
        callback = runtime.StepMetrics(log_dir=self.tmp / "adapter")
        for rate, memory in ((0.5, 7.5), (0.25, 8.1)):
            callback.on_train_loss_report(
                {"iteration": 1, "iterations_per_second": rate, "peak_memory": memory})
        out = json.loads((self.tmp / "adapter" / "train_metrics.json").read_text())
        self.assertEqual(out["peak_memory_gb"], 8.1)
        self.assertEqual(out["max_step_seconds"], 4.0)
        self.assertEqual(out["mean_step_seconds"], 3.0)


if __name__ == "__main__":
    unittest.main()
