"""Run the whole evaluation in two commands.

`run` generates base plus every adapter, then builds the blinded rating
sheets. Rate them with `python -m storythread.rate_ui`. `report` scores
whatever is rated. Both wrap the building blocks in eval.py.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import re

from . import eval as evaluation
from .data import rows

LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_DIR = "work/eval"


def add_commands(commands):
    """Registers the run and report subcommands on an argparse subparsers."""
    run = commands.add_parser(
        "run", help="generate every condition and build the rating sheets")
    run.add_argument("cases")
    run.add_argument("--adapter", action="append", default=[],
                     metavar="LABEL=PATH", help="repeat once per adapter")
    run.add_argument("--dir", default=DEFAULT_DIR)
    run.add_argument("--model", default="models/qwen35-4b-q4")
    run.add_argument("--max-tokens", type=int, default=160)
    run.add_argument("--seed", type=int, default=42)
    report = commands.add_parser("report", help="score the rated sheets")
    report.add_argument("cases")
    report.add_argument("--dir", default=DEFAULT_DIR)
    report.add_argument("--edges", default="4,12",
                        help="ascending turn-distance edges")


def parse_adapters(items):
    """Returns [(label, path)] from LABEL=PATH strings, base first."""
    conditions = [("base", None)]
    for item in items:
        label, _, path = item.partition("=")
        if not LABEL.match(label) or not path:
            raise SystemExit(f"--adapter expects LABEL=PATH, got {item!r}")
        if label in dict(conditions):
            raise SystemExit(f"Duplicate or reserved label: {label}")
        conditions.append((label, path))
    return conditions


def generate_condition(args, label, adapter, target):
    """Generates one condition into target, only replacing it when whole."""
    part = target.with_name(target.name + ".part")
    part.unlink(missing_ok=True)
    evaluation.generate(argparse.Namespace(
        model=args.model, adapter=adapter, label=label,
        max_tokens=args.max_tokens, cases=args.cases, output=part))
    os.replace(part, target)


def check_complete(target, cases, sha):
    """Raises SystemExit unless target answers exactly these cases."""
    done = evaluation.by_id(target)
    shas = {row.get("cases_sha") for row in done.values()}
    if done.keys() != cases.keys() or shas != {sha}:
        raise SystemExit(
            f"{target} answers different cases. Delete it to regenerate.")


def build_once(paths, same, build):
    """Runs build() when no path exists; raises if same() is False.

    A rating sheet holds hand work, so an existing one is never replaced.
    A partial set also raises, because same() cannot be trusted.
    """
    present = [path for path in paths if path.exists()]
    if not present:
        build()
        return "built"
    if len(present) < len(paths) or not same():
        raise SystemExit(
            "Existing sheets are incomplete or cover other conditions. Delete "
            + ", ".join(p.name for p in paths)
            + " to rebuild them (their ratings are lost).")
    return "kept"


def run(args):
    """Generates all conditions, then builds the continuity and blind sheets."""
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = evaluation.load_cases(args.cases)
    sha = evaluation.file_sha(args.cases)
    conditions = parse_adapters(args.adapter)
    labels = [label for label, _ in conditions]
    for label, adapter in conditions:
        target = out / f"{label}.jsonl"
        if target.exists():
            check_complete(target, cases, sha)
            print(f"{label}: already generated")
        else:
            print(f"{label}: generating")
            generate_condition(args, label, adapter, target)
    outputs = [out / f"{label}.jsonl" for label in labels]
    sheet, key = out / "sheet.jsonl", out / "key.jsonl"

    def same_sheet():
        shown = {c for row in rows(key) for c in row["labels"].values()}
        return shown == set(labels)

    print("continuity sheet:", build_once(
        [sheet, key], same_sheet, lambda: evaluation.sheet(argparse.Namespace(
            cases=args.cases, output=sheet, key=key, outputs=outputs,
            seed=args.seed))))
    for first, second in itertools.combinations(labels, 2):
        stem = f"blind-{first}-{second}"
        pair, pair_key = out / f"{stem}.jsonl", out / f"{stem}-key.jsonl"
        print(f"{stem}:", build_once(
            [pair, pair_key], lambda: True, lambda: evaluation.blind(
                argparse.Namespace(
                    cases=args.cases, first=out / f"{first}.jsonl",
                    second=out / f"{second}.jsonl", output=pair,
                    key=pair_key, seed=args.seed))))
    print(f"Next: python -m storythread.rate_ui --dir {out}")


def sheets(directory):
    """Returns the rating sheets in directory with rating progress.

    Each item has name, kind (continuity or blind), path, key, rated, total.
    """
    out = Path(directory)
    found = []
    if (out / "sheet.jsonl").is_file():
        marks = [r["satisfies"] for row in rows(out / "sheet.jsonl")
                 for r in row["replies"]]
        found.append({"name": "sheet", "kind": "continuity",
                      "path": out / "sheet.jsonl", "key": out / "key.jsonl",
                      "rated": sum(type(m) is bool for m in marks),
                      "total": len(marks)})
    for path in sorted(out.glob("blind-*.jsonl")):
        if path.name.endswith("-key.jsonl"):
            continue
        wins = [row["winner"] for row in rows(path)]
        found.append({"name": path.stem, "kind": "blind", "path": path,
                      "key": path.with_name(path.stem + "-key.jsonl"),
                      "rated": sum(w in ("A", "B", "tie") for w in wins),
                      "total": len(wins)})
    return found


def report(args):
    """Scores every fully rated sheet and reports progress on the rest."""
    out = Path(args.dir)
    result = {}
    for item in sheets(out):
        if item["rated"] < item["total"]:
            result[item["name"]] = {
                "status": f"{item['rated']} of {item['total']} rated"}
        elif item["kind"] == "continuity":
            result[item["name"]] = evaluation.continuity_report(
                argparse.Namespace(cases=args.cases, sheet=item["path"],
                                   key=item["key"], edges=args.edges))
        else:
            result[item["name"]] = evaluation.tally(item["path"], item["key"])
    if not result:
        raise SystemExit(f"No rating sheets in {out}. Run eval run first.")
    (out / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
