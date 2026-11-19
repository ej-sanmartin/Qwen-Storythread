"""Prepare a frozen MLX checkpoint and check the exact training objective."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from .data import rows, write_rows

DEFAULT_MODEL = "models/qwen35-4b-q4"
DEFAULT_REPO = "mlx-community/Qwen3.5-4B-4bit"
NON_THINKING = "{%- if enable_thinking is undefined -%}{%- set enable_thinking = false -%}{%- endif -%}\n"


def setup_model(args):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    target = Path(args.model)
    if target.exists():
        raise SystemExit(f"{target} already exists. Reuse it or select a new --model path.")
    snapshot = Path(snapshot_download(
        repo_id=args.repo, revision=args.revision,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.model", "*.txt", "*.tiktoken"]
    ))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
    template = tokenizer.chat_template
    if not isinstance(template, str) or "enable_thinking" not in template:
        raise SystemExit("This checkpoint does not expose the expected Qwen chat template.")
    target.mkdir(parents=True)
    for entry in snapshot.rglob("*"):
        if entry.is_file() and entry.name != "chat_template.jinja":
            mirror = target / entry.relative_to(snapshot)
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.symlink_to(entry.resolve())
    (target / "chat_template.jinja").write_text(NON_THINKING + template)
    provenance = {"repo": args.repo, "revision": snapshot.name, "template": "non-thinking default"}
    (target / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))


def encode(tokenizer, messages, prompt=False):
    """Returns chat-template token IDs; prompt=True appends the reply header."""
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=prompt, return_dict=False)


def measure(tokenizer, messages, evidence_turn=None):
    """Returns token counts for one example and, if given, its evidence span.

    Input is everything before the supervised reply. Evidence offsets are
    approximate to a few template tokens. Distance runs from the evidence turn
    start to the start of the supervised reply.
    """
    full = encode(tokenizer, messages)
    prefix = encode(tokenizer, messages[:-1], prompt=True)
    target = full[len(prefix):]
    info = {"full": len(full), "input": len(prefix), "target": len(target),
            "aligned": messages[-1]["role"] == "assistant" and full[:len(prefix)] == prefix,
            "empty": not tokenizer.decode(target, skip_special_tokens=True).strip()}
    if evidence_turn is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        cursor = start = 0
        for message in messages[:evidence_turn + 2]:  # messages[0] is the system card
            start = text.find(message["content"], cursor)
            cursor = start + len(message["content"])

        def count(end):
            return len(tokenizer.encode(text[:end], add_special_tokens=False))

        info.update(evidence_end=count(cursor),
                    evidence_distance_tokens=len(prefix) - count(start))
    return info


def flags_for(info, cap):
    """Returns reasons this row would lose evidence, its target, or its mask."""
    flags = []
    if not info["aligned"]:
        flags.append("mask_misaligned")
    if info["empty"]:
        flags.append("empty_target")
    if info["full"] > cap:
        flags.append("target_truncated")
    if info.get("evidence_end", 0) > cap:
        flags.append("evidence_truncated")
    return flags


def stats(values):
    """Returns n/min/median/p95/max for a list of numbers, or None."""
    values = sorted(values)
    if not values:
        return None
    p95 = values[math.ceil(0.95 * len(values)) - 1]
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            "p95": p95, "max": values[-1]}


def summarize(results):
    """Groups lengths and evidence distances by source and branch."""
    groups = {}
    for item in results:
        groups.setdefault(f"{item['source']}/{item['branch'] or '-'}", []).append(item)
    return {name: {"examples": len(items),
                   "input_tokens": stats([i["input"] for i in items]),
                   "target_tokens": stats([i["target"] for i in items]),
                   "full_tokens": stats([i["full"] for i in items]),
                   "evidence_distance_turns": stats([i["distance_turns"] for i in items
                                                     if i.get("distance_turns") is not None]),
                   "evidence_distance_tokens": stats([i["evidence_distance_tokens"] for i in items
                                                      if "evidence_distance_tokens" in i])}
            for name, items in groups.items()}


def preflight(data_dir, tokenizer, cap):
    """Returns {split: {"summary": ..., "flagged": [...]}} for a dataset folder."""
    folder = Path(data_dir)
    manifest = {}
    if (folder / "manifest.jsonl").exists():
        manifest = {(m["split"], m["row"]): m for m in rows(folder / "manifest.jsonl")}
    report = {}
    for split in ("train", "valid"):
        results = []
        for number, row in enumerate(rows(folder / f"{split}.jsonl")):
            meta = manifest.get((split, number), {})
            info = measure(tokenizer, row["messages"], meta.get("evidence_turn"))
            info.update(split=split, row=number, source=meta.get("source", "unknown"),
                        branch=meta.get("branch"),
                        distance_turns=meta.get("evidence_distance_turns"))
            info["flags"] = flags_for(info, cap)
            results.append(info)
        if not results:
            raise SystemExit(f"{folder / (split + '.jsonl')} is empty")
        flagged = [{"row": r["row"], "source": r["source"], "full": r["full"],
                    "flags": r["flags"]} for r in results if r["flags"]]
        report[split] = {"summary": summarize(results), "flagged": flagged}
    return report


def check_data(args):
    from transformers import AutoTokenizer

    report = preflight(args.data, AutoTokenizer.from_pretrained(args.model), args.max_tokens)
    report["max_tokens"] = args.max_tokens
    out = Path(args.data) / f"preflight-{args.max_tokens}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    flagged = sum(len(report[s]["flagged"]) for s in ("train", "valid"))
    print(f"Wrote {out}; {flagged} flagged row(s) at cap {args.max_tokens}")
    if flagged and not args.report_only:
        raise SystemExit(1)


def probe(args):
    """Writes a tiny dataset of the longest train examples that fit the target."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sized = []
    for row in rows(Path(args.data) / "train.jsonl"):
        size = len(encode(tokenizer, row["messages"]))
        if size <= args.target_tokens:
            sized.append((size, row))
    sized.sort(key=lambda pair: -pair[0])
    chosen = sized[:args.count]
    if not chosen or chosen[0][0] < 0.6 * args.target_tokens:
        raise SystemExit(f"No example near {args.target_tokens} tokens; "
                         f"longest fit is {chosen[0][0] if chosen else 'none'}. "
                         "Author a longer scene first.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "train.jsonl", (row for _, row in chosen))
    write_rows(out / "valid.jsonl", (row for _, row in chosen[:1]))
    print(json.dumps({"output": str(out), "lengths": [size for size, _ in chosen]}))


class StepMetrics:
    """mlx-lm training callback that records step time and peak memory.

    Writes <adapter_path>/train_metrics.json after every training report.
    """

    def __init__(self, project_name=None, log_dir=None, config=None, wrapped_callback=None):
        self.path = Path(log_dir) / "train_metrics.json"
        self.reports = []

    def on_train_loss_report(self, info):
        self.reports.append(info)
        steps = [1 / r["iterations_per_second"] for r in self.reports]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "last_iter": info["iteration"],
            "mean_step_seconds": round(statistics.mean(steps), 3),
            "max_step_seconds": round(max(steps), 3),
            "peak_memory_gb": round(max(r["peak_memory"] for r in self.reports), 3),
        }, indent=2) + "\n")

    def on_val_loss_report(self, info):
        pass


def train(args):
    """Runs mlx_lm.lora with its own config handling, plus StepMetrics."""
    from mlx_lm import lora
    from mlx_lm.tuner import callbacks

    callbacks.SUPPORT_CALLBACK["stepmetrics"] = StepMetrics
    passthrough = [a for a in args.lora_args if a != "--"]
    sys.argv = ["mlx_lm.lora", *passthrough, "--report-to", "stepmetrics"]
    lora.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup-model")
    setup.add_argument("--repo", default=DEFAULT_REPO)
    setup.add_argument("--revision")
    check = commands.add_parser("check-data")
    check.add_argument("--data", required=True)
    check.add_argument("--max-tokens", type=int, default=512)
    check.add_argument("--report-only", action="store_true")
    probing = commands.add_parser("probe")
    probing.add_argument("--data", required=True)
    probing.add_argument("--target-tokens", type=int, required=True)
    probing.add_argument("--count", type=int, default=4)
    probing.add_argument("--output", required=True)
    training = commands.add_parser("train")
    training.add_argument("lora_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    {"setup-model": setup_model, "check-data": check_data, "probe": probe,
     "train": train}[args.command](args)


if __name__ == "__main__":
    main()
