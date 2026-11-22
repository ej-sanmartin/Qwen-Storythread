"""Checks that an ablation plan zeroes only the named lora_b tensors.

Run: python -m unittest discover tests
"""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, ROOT.parent.as_posix())
sys.path.insert(0, ROOT.as_posix())

from storythread import ablate  # noqa: E402
from storythread import inspect_adapters as ia  # noqa: E402
from test_inspect_adapters import PREFIX, make_run, write_safetensors  # noqa: E402


def payload(path, key):
    """Returns (dtype, shape, raw bytes) for one tensor in a safetensors file."""
    data = Path(path).read_bytes()
    (size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + size])
    info = header[key]
    begin, end = info["data_offsets"]
    return info["dtype"], info["shape"], data[8 + size + begin:8 + size + end]


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.folder = make_run(self.tmp / "run", [25], seed=4)
        self.source = self.folder / "0000025_adapters.safetensors"
        self.before = self.source.read_bytes()

    def plan(self, modules, digest=None):
        record = ablate.checkpoint_record(self.source)
        body = {"schema": 1, "arm": "recent", "checkpoint": record["path"],
                "sha256": digest or record["sha256"], "modules": modules}
        path = self.tmp / "plan.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def test_zeroes_one_projection_and_not_the_same_name_elsewhere(self):
        named = PREFIX.format(0, "up_proj")
        twin = PREFIX.format(1, "up_proj")
        out = self.tmp / "out"
        keys = ablate.apply_plan(self.plan([named, named]), out)
        self.assertEqual(keys, [f"{named}.lora_b"])
        self.assertEqual(self.source.read_bytes(), self.before)
        dtype, shape, raw = payload(out / "adapters.safetensors", f"{named}.lora_b")
        self.assertEqual(dtype, "F32")
        self.assertEqual(shape, [2, 9])
        self.assertEqual(raw, b"\x00" * len(raw))
        self.assertNotEqual(payload(self.source, f"{named}.lora_b")[2], raw)
        for key in (f"{named}.lora_a", f"{twin}.lora_b", f"{twin}.lora_a"):
            self.assertEqual(payload(out / "adapters.safetensors", key),
                             payload(self.source, key))
        config = (out / "adapter_config.json").read_bytes()
        self.assertEqual(config, (self.folder / "adapter_config.json").read_bytes())

    def test_whole_block_leaves_the_other_block(self):
        modules = [PREFIX.format(0, name) for name in ("up_proj", "down_proj")]
        out = self.tmp / "block"
        ablate.apply_plan(self.plan(modules), out)
        for name in ("up_proj", "down_proj"):
            zeroed = PREFIX.format(0, name) + ".lora_b"
            kept = PREFIX.format(1, name) + ".lora_b"
            _, _, raw = payload(out / "adapters.safetensors", zeroed)
            self.assertEqual(raw, b"\x00" * len(raw))
            self.assertEqual(payload(out / "adapters.safetensors", kept),
                             payload(self.source, kept))

    def test_f16_dtype_and_shape_survive(self):
        path = self.tmp / "f16" / "0000001_adapters.safetensors"
        path.parent.mkdir()
        module = "model.layers.2.mlp.up_proj"
        tensors = {f"{module}.lora_a": np.ones((4, 2)),
                   f"{module}.lora_b": np.ones((2, 3))}
        write_safetensors(path, tensors, kind="F16")
        (path.parent / "adapter_config.json").write_text("{}", encoding="utf-8")
        plan = self.tmp / "f16.json"
        record = ablate.checkpoint_record(path)
        plan.write_text(json.dumps({
            "schema": 1, "arm": "recent", "checkpoint": record["path"],
            "sha256": record["sha256"], "modules": [module]}), encoding="utf-8")
        out = self.tmp / "f16-out"
        ablate.apply_plan(plan, out)
        dtype, shape, raw = payload(out / "adapters.safetensors", f"{module}.lora_b")
        self.assertEqual((dtype, shape, raw), ("F16", [2, 3], b"\x00" * len(raw)))
        self.assertEqual(payload(out / "adapters.safetensors", f"{module}.lora_a")[0],
                         "F16")

    def test_rejects_a_hash_mismatch_without_writing(self):
        out = self.tmp / "bad"
        with self.assertRaises(ValueError):
            ablate.apply_plan(self.plan([PREFIX.format(0, "up_proj")], "0" * 64), out)
        self.assertFalse(out.exists())
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_rejects_an_existing_output_directory(self):
        out = self.tmp / "out"
        out.mkdir()
        with self.assertRaises(ValueError):
            ablate.apply_plan(self.plan([PREFIX.format(0, "up_proj")]), out)
        self.assertEqual(list(out.iterdir()), [])
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_rejects_a_module_that_is_not_in_the_checkpoint(self):
        out = self.tmp / "missing"
        with self.assertRaises(ValueError):
            ablate.apply_plan(self.plan(["model.layers.9.mlp.up_proj"]), out)
        self.assertFalse(out.exists())

    def test_cli_writes_the_adapter_directory(self):
        out = self.tmp / "cli"
        ia.main(["--plan", self.plan([PREFIX.format(0, "down_proj")]).as_posix(),
                 "--output", out.as_posix()])
        self.assertTrue((out / "adapters.safetensors").is_file())
        _, _, raw = payload(out / "adapters.safetensors",
                            PREFIX.format(0, "down_proj") + ".lora_b")
        self.assertEqual(raw, b"\x00" * len(raw))

    def test_report_records_the_checkpoint_hash(self):
        report = ia.build_report({"recent": self.folder, "earlier": self.folder})
        saved = report["checkpoints"]["recent"]["25"]
        self.assertEqual(saved["sha256"], ablate.checkpoint_record(self.source)["sha256"])
        self.assertTrue(saved["path"].endswith("0000025_adapters.safetensors"))
