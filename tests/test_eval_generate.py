"""Checks how `eval generate` builds prompts, with MLX replaced by stubs.

Run: python -m unittest discover tests
"""
import argparse
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from storythread import eval as evaluation  # noqa: E402
from storythread.data import rows  # noqa: E402
from test_fixture import FIXTURES  # noqa: E402


class RecordingTokenizer:
    """Records the chat-template arguments of every prompt it builds."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return [1, 2, 3]

    def encode(self, text, add_special_tokens=False):
        return text.split()


def stub_modules(tokenizer):
    """Returns sys.modules entries that stand in for mlx and mlx_lm."""
    core = types.ModuleType("mlx.core")
    core.synchronize = lambda: None
    package = types.ModuleType("mlx")
    package.core = core
    lm = types.ModuleType("mlx_lm")
    lm.load = lambda model, adapter_path=None: (object(), tokenizer)
    lm.generate = lambda *args, **kwargs: "reply"
    sample = types.ModuleType("mlx_lm.sample_utils")
    sample.make_sampler = lambda temp: None
    lm.sample_utils = sample
    return {"mlx": package, "mlx.core": core, "mlx_lm": lm,
            "mlx_lm.sample_utils": sample}


class GenerateTest(unittest.TestCase):
    def test_prompts_turn_thinking_off(self):
        tokenizer = RecordingTokenizer()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                sys.modules, stub_modules(tokenizer)):
            output = Path(tmp) / "out.jsonl"
            evaluation.generate(argparse.Namespace(
                model=tmp, adapter=None, label="base", max_tokens=8,
                cases=(FIXTURES / "eval-test.jsonl").as_posix(),
                output=output))
            written = list(rows(output))
        self.assertTrue(tokenizer.calls)
        self.assertEqual(len(tokenizer.calls), len(written))
        for call in tokenizer.calls:
            self.assertIs(call["enable_thinking"], False)
            self.assertTrue(call["add_generation_prompt"])


if __name__ == "__main__":
    unittest.main()
