"""Prepare a frozen MLX checkpoint and check the exact training objective."""
import argparse
import json
from pathlib import Path

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


def check_data(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for split in ("train", "valid"):
        path = Path(args.data) / f"{split}.jsonl"
        count = longest = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
            prefix = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if messages[-1]["role"] != "assistant" or full[:len(prefix)] != prefix:
                raise SystemExit(f"{path}, record {count + 1}: assistant mask does not align")
            supervised = full[len(prefix):]
            sample = tokenizer.decode(supervised, skip_special_tokens=True)
            if not sample.strip() or len(full) > args.max_tokens:
                raise SystemExit(f"{path}, record {count + 1}: {len(full)} tokens or empty target")
            count += 1
            longest = max(longest, len(full))
        if not count:
            raise SystemExit(f"{path} is empty")
        print(f"{path}: {count} records; longest {longest} tokens")


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
    args = parser.parse_args()
    if args.command == "setup-model":
        setup_model(args)
    else:
        check_data(args)


if __name__ == "__main__":
    main()
