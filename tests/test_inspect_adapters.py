"""Checks the adapter inspection report on tiny synthetic adapters.

Run: python -m unittest discover tests
"""
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("numpy is not installed")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from storythread import inspect_adapters as ia  # noqa: E402

PREFIX = "language_model.model.layers.{}.mlp.{}"


def write_safetensors(path, tensors, kind="F32"):
    """Writes {name: float32 array} as a safetensors file of dtype kind."""
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        array = np.asarray(array, dtype=np.float32)
        if kind == "BF16":
            blob = (array.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        else:
            blob = array.astype(np.float16 if kind == "F16" else np.float32
                                ).tobytes()
        header[name] = {"dtype": kind, "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    text = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def adapter_tensors(seed, blocks=(0, 1), rank=2):
    """Returns random lora_a/lora_b tensors for two projections per block."""
    rng = np.random.default_rng(seed)
    tensors = {}
    for block in blocks:
        for name, (n_in, n_out) in (("up_proj", (6, 9)), ("down_proj", (9, 6))):
            module = PREFIX.format(block, name)
            tensors[f"{module}.lora_a"] = rng.normal(size=(n_in, rank))
            tensors[f"{module}.lora_b"] = rng.normal(size=(rank, n_out))
    return {k: v.astype(np.float32) for k, v in tensors.items()}


def make_run(folder, steps, seed, scale=2.0, final=True, last_iter=None):
    """Writes a fake MLX-LM run directory and returns its path."""
    folder = Path(folder)
    folder.mkdir(parents=True)
    config = {"fine_tune_type": "lora", "lora_parameters": {"scale": scale}}
    (folder / "adapter_config.json").write_text(json.dumps(config))
    for step in steps:
        write_safetensors(folder / f"{step:07d}_adapters.safetensors",
                          adapter_tensors(seed + step))
    if final:
        last = folder / f"{max(steps):07d}_adapters.safetensors"
        (folder / "adapters.safetensors").write_bytes(last.read_bytes())
    if last_iter is not None:
        (folder / "train_metrics.json").write_text(
            json.dumps({"last_iter": last_iter}))
    return folder


class RmsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.a = rng.normal(size=(7, 3)).astype(np.float32)
        self.b = rng.normal(size=(3, 5)).astype(np.float32)

    def dense(self, scale):
        delta = scale * (self.a.astype(np.float64) @ self.b.astype(np.float64))
        return float(np.sqrt(np.mean(delta ** 2)))

    def test_matches_dense_reference(self):
        self.assertAlmostEqual(ia.effective_rms(self.a, self.b, 2.0),
                               self.dense(2.0), places=9)

    def test_negative_scale_uses_magnitude(self):
        self.assertAlmostEqual(ia.effective_rms(self.a, self.b, -2.0),
                               self.dense(2.0), places=9)

    def test_rebalancing_factors_keeps_rms(self):
        base = ia.effective_rms(self.a, self.b, 2.0)
        scaled = ia.effective_rms(self.a * 2, self.b / 2, 2.0)
        self.assertAlmostEqual(base, scaled, places=9)

    def test_matches_mlx_lora_linear(self):
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm.tuner.lora import LoRALinear
        except ImportError:
            self.skipTest("mlx_lm is unavailable (not installed or no Metal)")
        base = nn.Linear(6, 5)
        layer = LoRALinear.from_base(base, r=3, scale=2.0)
        layer.lora_a = mx.random.normal(layer.lora_a.shape)
        layer.lora_b = mx.random.normal(layer.lora_b.shape)
        eye = mx.eye(6)
        delta = np.array(layer(eye) - base(eye))
        expected = float(np.sqrt(np.mean(delta ** 2)))
        got = ia.effective_rms(np.array(layer.lora_a), np.array(layer.lora_b),
                               2.0)
        self.assertAlmostEqual(got, expected, places=4)


class MeasureTest(unittest.TestCase):
    def test_incomplete_pair_is_an_error(self):
        tensors = {"layers.0.mlp.up_proj.lora_a": np.ones((4, 2), np.float32)}
        with self.assertRaisesRegex(ValueError, "no lora_b"):
            ia.measure(tensors, 2.0, "x.safetensors")

    def test_incompatible_rank_is_an_error(self):
        tensors = {"layers.0.mlp.up_proj.lora_a": np.ones((4, 2), np.float32),
                   "layers.0.mlp.up_proj.lora_b": np.ones((3, 5), np.float32)}
        with self.assertRaisesRegex(ValueError, "share a rank"):
            ia.measure(tensors, 2.0, "x.safetensors")

    def test_non_lora_tensor_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "not a lora_a"):
            ia.measure({"layers.0.bias": np.ones(3, np.float32)}, 2.0, "x")

    def test_reports_dimensions_and_rank(self):
        stats = ia.measure(adapter_tensors(0), 2.0, "x")
        entry = stats[PREFIX.format(0, "up_proj")]
        self.assertEqual((entry["in"], entry["out"], entry["rank"]), (6, 9, 2))


class ReaderTest(unittest.TestCase):
    def test_bf16_and_f16_round_trip(self):
        values = {"w": np.array([[1.5, -2.25], [0.5, 8.0]], np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("BF16", "F16", "F32"):
                path = Path(tmp) / f"{kind}.safetensors"
                write_safetensors(path, values, kind)
                np.testing.assert_array_equal(
                    ia.read_safetensors(path)["w"], values["w"])

    def test_rejects_garbage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.safetensors"
            path.write_bytes(b"not safetensors")
            with self.assertRaisesRegex(ValueError, "not a readable"):
                ia.read_safetensors(path)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_arm_missing_a_step_is_reported(self):
        recent = make_run(self.tmp / "recent", [25, 50, 100], seed=1)
        earlier = make_run(self.tmp / "earlier", [25, 100], seed=2)
        report = ia.build_report({"recent": recent, "earlier": earlier})
        self.assertEqual(report["steps"], [25, 50, 100])
        self.assertEqual(report["missing"],
                         {"recent": [], "earlier": [50]})
        self.assertNotIn("50", report["rms"]["earlier"])
        self.assertIn("50", report["rms"]["recent"])

    def test_final_duplicate_is_shown_once(self):
        folder = make_run(self.tmp / "run", [25, 50], seed=1)
        steps, notes = ia.find_checkpoints(folder)
        self.assertEqual(list(steps), [25, 50])
        self.assertEqual(notes, [])

    def test_distinct_final_uses_recorded_step(self):
        folder = make_run(self.tmp / "run", [25, 50], seed=1, last_iter=60)
        write_safetensors(folder / "adapters.safetensors", adapter_tensors(9))
        steps, _ = ia.find_checkpoints(folder)
        self.assertEqual(list(steps), [25, 50, 60])

    def test_distinct_final_without_step_is_skipped_with_note(self):
        folder = make_run(self.tmp / "run", [25, 50], seed=1)
        write_safetensors(folder / "adapters.safetensors", adapter_tensors(9))
        steps, notes = ia.find_checkpoints(folder)
        self.assertEqual(list(steps), [25, 50])
        self.assertEqual(len(notes), 1)

    def test_scale_comes_from_each_config(self):
        recent = make_run(self.tmp / "recent", [25], seed=1, scale=2.0)
        earlier = make_run(self.tmp / "earlier", [25], seed=1, scale=4.0)
        report = ia.build_report({"recent": recent, "earlier": earlier})
        path = PREFIX.format(0, "up_proj")
        ratio = (report["rms"]["earlier"]["25"][path]
                 / report["rms"]["recent"]["25"][path])
        self.assertAlmostEqual(ratio, 2.0, places=4)

    def test_report_layout(self):
        recent = make_run(self.tmp / "recent", [25], seed=1)
        earlier = make_run(self.tmp / "earlier", [25], seed=2)
        report = ia.build_report({"recent": recent, "earlier": earlier})
        self.assertEqual(report["blocks"], [0, 1])
        self.assertEqual([c["label"] for c in report["columns"]],
                         ["down_proj", "up_proj"])
        self.assertEqual(len(report["modules"]), 4)

    def test_cli_writes_standalone_page(self):
        recent = make_run(self.tmp / "recent", [25, 50], seed=1)
        earlier = make_run(self.tmp / "earlier", [25, 50], seed=2)
        output = self.tmp / "out" / "heatmap.html"
        ia.main(["--recent", recent.as_posix(), "--earlier",
                 earlier.as_posix(), "--output", output.as_posix()])
        page = output.read_text(encoding="utf-8")
        self.assertIn("Effective LoRA update RMS", page)
        self.assertIn("separate question for ablation", page)
        for placeholder in ("{{style}}", "{{script}}", "{{data}}"):
            self.assertNotIn(placeholder, page)
        data = re.search(r'id="report">(.*?)</script>', page, re.S)[1]
        self.assertEqual(json.loads(data)["steps"], [25, 50])


if __name__ == "__main__":
    unittest.main()
