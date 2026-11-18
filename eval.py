"""Generate controlled next replies, prepare blind comparisons, and count preferences."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

from data import rows, write_rows


def by_id(path):
    result = {}
    for row in rows(path):
        if row["id"] in result:
            raise ValueError(f"Duplicate ID in {path}: {row['id']}")
        result[row["id"]] = row
    return result


def generate(args):
    import mlx.core as mx
    from mlx_lm import generate as reply
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    cases = by_id(args.cases)
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.0)

    def run():
        for case_id, case in cases.items():
            messages = case["messages"]
            if messages[-1]["role"] != "user":
                raise ValueError(f"Case {case_id} must end with a user message")
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            start = perf_counter()
            output = reply(model, tokenizer, prompt=tokens, sampler=sampler,
                           max_tokens=args.max_tokens, verbose=False)
            mx.synchronize()
            yield {"id": case_id, "output": output, "condition": args.label,
                   "model": args.model, "adapter": args.adapter,
                   "prompt_sha": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                   "prompt_tokens": len(tokens),
                   "output_tokens_reencoded": len(tokenizer.encode(output, add_special_tokens=False)),
                   "max_new_tokens": args.max_tokens,
                   "elapsed_seconds": round(perf_counter() - start, 3)}

    write_rows(args.output, run())


def blind(args):
    cases, first, second = (by_id(path) for path in (args.cases, args.first, args.second))
    if cases.keys() != first.keys() or first.keys() != second.keys():
        raise ValueError("Comparison requires exactly the same IDs")
    public, key = [], []
    rng = random.Random(args.seed)
    for case_id, case in cases.items():
        left, right = first[case_id], second[case_id]
        if left["condition"] == right["condition"]:
            raise ValueError("Use different labels for the compared conditions")
        if (left["prompt_sha"], left["max_new_tokens"]) != (right["prompt_sha"], right["max_new_tokens"]):
            raise ValueError(f"Prompts or generation budgets differ for {case_id}")
        if rng.randrange(2):
            left, right = right, left
        public.append({"id": case_id, "messages": case["messages"],
                       "checks": case.get("checks", []), "A": left["output"], "B": right["output"],
                       "winner": "", "comment": ""})
        key.append({"id": case_id, "A": left["condition"], "B": right["condition"]})
    write_rows(args.output, public)
    write_rows(args.key, key)


def score(args):
    decisions, key = by_id(args.ratings), by_id(args.key)
    if decisions.keys() != key.keys():
        raise ValueError("Rate every case exactly once.")
    counts = {"tie": 0}
    for case_id, row in decisions.items():
        choice = row["winner"]
        if choice not in {"A", "B", "tie"}:
            raise ValueError(f"Case {case_id}: winner must be A, B, or tie.")
        for side in ("A", "B"):
            counts.setdefault(key[case_id][side], 0)
        winner = "tie" if choice == "tie" else key[case_id][choice]
        counts[winner] = counts.get(winner, 0) + 1
    print(json.dumps({"cases": len(decisions), "preferred": counts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("generate")
    run.add_argument("--model", default="models/qwen35-4b-q4")
    run.add_argument("--adapter")
    run.add_argument("--label", required=True)
    run.add_argument("--max-tokens", type=int, default=160)
    run.add_argument("cases")
    run.add_argument("output")
    compare = commands.add_parser("blind")
    for field in ("cases", "first", "second", "output", "key"):
        compare.add_argument(field)
    compare.add_argument("--seed", type=int, default=42)
    summarize = commands.add_parser("score")
    summarize.add_argument("ratings")
    summarize.add_argument("key")
    args = parser.parse_args()
    {"generate": generate, "blind": blind, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
