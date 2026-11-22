"""Generate controlled replies, build blind rating sheets, and score them.

Primary metric: percent of scenarios where a model acts on the correct story
state in both versions (event-triggered design after REMP, Yu et al., 2026).
Engagement preference is a separate pairwise score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

from .data import rows, write_rows
from .scenarios import VERSIONS, check_disjoint, parse_scenarios

QUESTION = "Which character would you rather keep chatting with?"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def by_id(path):
    result = {}
    for row in rows(path):
        if row["id"] in result:
            raise ValueError(f"Duplicate ID in {path}: {row['id']}")
        result[row["id"]] = row
    return result


def load_cases(path):
    """Returns {case_id: case}. Scenario files expand to <scenario_id>/<a|b>.

    Legacy rows that already carry an id and messages pass through unchanged.
    """
    data = list(rows(path))
    if not data or "scenario_id" not in data[0]:
        return by_id(path)
    cases = {}
    for scenario in parse_scenarios(data):
        for name in VERSIONS:
            version = scenario["versions"][name]
            cases[f"{scenario['scenario_id']}/{name}"] = {
                "id": f"{scenario['scenario_id']}/{name}",
                "scenario_id": scenario["scenario_id"], "version": name,
                "distance": scenario["distance"], **version}
    return cases


def generate(args):
    import mlx.core as mx
    from mlx_lm import generate as reply
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    cases = load_cases(args.cases)
    cases_sha = file_sha(args.cases)
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.0)
    source = Path(args.model) / "source.json"
    revision = json.loads(source.read_text()).get("revision") if source.exists() else None

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
                   "model": args.model, "model_revision": revision,
                   "adapter": args.adapter, "cases_sha": cases_sha,
                   "prompt_sha": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                   "prompt_tokens": len(tokens),
                   "output_tokens_reencoded": len(tokenizer.encode(output, add_special_tokens=False)),
                   "max_new_tokens": args.max_tokens, "temperature": 0.0,
                   "elapsed_seconds": round(perf_counter() - start, 3)}

    write_rows(args.output, run())


def same_setup(cases, runs):
    """Raises unless every run answers the same cases under the same settings."""
    conditions = []
    for run in runs:
        if run.keys() != cases.keys():
            raise ValueError("Comparison requires exactly the same IDs")
        conditions.append({row["condition"] for row in run.values()})
    if any(len(c) != 1 for c in conditions) or len(set().union(*conditions)) != len(runs):
        raise ValueError("Each output file needs one distinct condition label")
    for case_id in cases:
        setups = {(r[case_id]["prompt_sha"], r[case_id]["max_new_tokens"],
                   r[case_id].get("cases_sha")) for r in runs}
        if len(setups) != 1:
            raise ValueError(f"Prompts, budgets, or case files differ for {case_id}")
    return [next(iter(c)) for c in conditions]


def blind(args):
    """Pairs two conditions anonymously for the engagement question."""
    cases = load_cases(args.cases)
    first, second = by_id(args.first), by_id(args.second)
    same_setup(cases, [first, second])
    public, key = [], []
    rng = random.Random(args.seed)
    for case_id, case in cases.items():
        left, right = first[case_id], second[case_id]
        if rng.randrange(2):
            left, right = right, left
        public.append({"id": case_id, "question": QUESTION, "messages": case["messages"],
                       "A": left["output"], "B": right["output"],
                       "winner": "", "comment": ""})
        key.append({"id": case_id, "A": left["condition"], "B": right["condition"]})
    write_rows(args.output, public)
    write_rows(args.key, key)


def tally(ratings, key_path):
    """Returns blind engagement preference counts per condition."""
    decisions, key = by_id(ratings), by_id(key_path)
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
    return {"cases": len(decisions), "preferred": counts}


def score(args):
    """Prints blind engagement preference counts per condition."""
    print(json.dumps(tally(args.ratings, args.key), indent=2))


def validate(args):
    """Checks scenario files and that development and test sets are disjoint."""
    dev = parse_scenarios(rows(args.dev))
    test = parse_scenarios(rows(args.test))
    check_disjoint(dev, test)
    print(json.dumps({"dev": len(dev), "test": len(test), "test_sha": file_sha(args.test)}))


def sheet(args):
    """Writes an anonymized sheet where the reviewer marks each reply."""
    cases = load_cases(args.cases)
    if not all("expected" in c for c in cases.values()):
        raise ValueError("Continuity sheets need a scenario file with expected behaviors.")
    runs = [by_id(path) for path in args.outputs]
    conditions = same_setup(cases, runs)
    rng = random.Random(args.seed)
    public, key = [], []
    for case_id, case in cases.items():
        order = list(range(len(runs)))
        rng.shuffle(order)
        labels = {f"R{n + 1}": conditions[i] for n, i in enumerate(order)}
        replies = [{"label": f"R{n + 1}", "text": runs[i][case_id]["output"],
                    "satisfies": None, "comment": ""} for n, i in enumerate(order)]
        public.append({"id": case_id, "scenario_id": case["scenario_id"],
                       "messages": case["messages"], "expected": case["expected"],
                       "evidence_turn": case["evidence_turn"], "evidence": case["evidence"],
                       "replies": replies})
        key.append({"id": case_id, "labels": labels})
    write_rows(args.output, public)
    write_rows(args.key, key)


def bucket_labels(edges):
    """Returns ordered distance-bucket labels for ascending turn edges."""
    labels, low = [], 0
    for edge in edges:
        labels.append(f"<={edge}" if not low else f"{low + 1}-{edge}")
        low = edge
    return labels + [f">{edges[-1]}"]


def cell(count, total):
    return {"count": count, "total": total,
            "percent": round(100 * count / total, 1) if total else None}


def continuity_report(args):
    """Returns, per condition, how often both versions satisfy expectations."""
    cases, sheet_rows, key = load_cases(args.cases), by_id(args.sheet), by_id(args.key)
    if not (cases.keys() == sheet_rows.keys() == key.keys()):
        raise ValueError("Sheet, key, and cases must cover the same IDs.")
    edges = [int(e) for e in args.edges.split(",")]
    labels = bucket_labels(edges)
    verdicts = {}
    for case_id, row in sheet_rows.items():
        for reply in row["replies"]:
            if type(reply["satisfies"]) is not bool:
                raise ValueError(f"{case_id} {reply['label']}: set satisfies to true or false.")
            condition = key[case_id]["labels"][reply["label"]]
            verdicts.setdefault(condition, {})[case_id] = reply["satisfies"]
    result = {}
    for condition, marks in verdicts.items():
        scenarios = {}
        for case_id, ok in marks.items():
            scenarios.setdefault(cases[case_id]["scenario_id"], {})[cases[case_id]["version"]] = ok
        ranges = {label: [0, 0] for label in labels}
        for scenario_id, versions in scenarios.items():
            distance = cases[f"{scenario_id}/a"]["distance"]
            label = next((l for l, e in zip(labels, edges) if distance <= e), labels[-1])
            ranges[label][0] += all(versions.values())
            ranges[label][1] += 1
        result[condition] = {
            "both_versions": cell(sum(c for c, _ in ranges.values()), len(scenarios)),
            "single_versions": cell(sum(marks.values()), len(marks)),
            "by_evidence_distance_turns": {l: cell(*v) for l, v in ranges.items()}}
    return result


def continuity(args):
    """Prints the continuity report."""
    print(json.dumps(continuity_report(args), indent=2))


def main():
    from . import eval_flow  # imported late: eval_flow imports this module
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
    check = commands.add_parser("validate")
    check.add_argument("dev")
    check.add_argument("test")
    rate = commands.add_parser("sheet")
    rate.add_argument("cases")
    rate.add_argument("output")
    rate.add_argument("key")
    rate.add_argument("--outputs", nargs="+", required=True)
    rate.add_argument("--seed", type=int, default=42)
    primary = commands.add_parser("continuity")
    for field in ("cases", "sheet", "key"):
        primary.add_argument(field)
    primary.add_argument("--edges", default="4,12", help="ascending turn-distance edges")
    eval_flow.add_commands(commands)
    args = parser.parse_args()
    {"generate": generate, "blind": blind, "score": score, "validate": validate,
     "sheet": sheet, "continuity": continuity, "run": eval_flow.run,
     "report": eval_flow.report}[args.command](args)


if __name__ == "__main__":
    main()
