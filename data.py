"""Index PIPPA and export human-reviewed continuations to MLX-LM."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def rows(path):
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write_rows(path, data):
    with Path(path).open("x", encoding="utf-8") as file:
        for row in data:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def clean(value):
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").strip()


def records(path):
    for row in rows(path):
        turns = []
        for turn in row.get("conversation", []):
            speech = clean(turn.get("message"))
            human = turn.get("is_human")
            if not speech or type(human) is not bool:
                continue
            role = "user" if human else "assistant"
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n\n" + speech
            else:
                turns.append({"role": role, "content": speech})
        if len(turns) < 3:
            continue
        name = clean(row.get("bot_name"))
        description = clean(row.get("bot_description"))
        greeting = clean(row.get("bot_greeting"))
        if greeting and turns[0]["role"] == "user":
            turns.insert(0, {"role": "assistant", "content": greeting})
        group = " ".join(re.findall(r"\w+", name.casefold()))
        scene = {"name": name, "description": description, "turns": turns}
        yield fingerprint(scene), group, scene


def index(args):
    pattern = re.compile(args.pattern, re.IGNORECASE) if args.pattern else None
    seen = set()

    def candidates():
        for scene_id, group, scene in records(args.source):
            if scene_id in seen:
                continue
            text = " ".join([scene["name"], scene["description"], *(t["content"] for t in scene["turns"])])
            if pattern and not pattern.search(text):
                continue
            seen.add(scene_id)
            yield {"scene_id": scene_id, "suggested_group": group, **scene}

    write_rows(args.output, candidates())


def prepare(args):
    review = {}
    groups = {}
    for row in rows(args.review):
        scene_id, group, split = row["scene_id"], row["group"], row["split"]
        if not isinstance(group, str) or not group.strip() or split not in {"train", "valid"}:
            raise ValueError("Review requires a nonempty group and a train or valid split.")
        group = group.strip().casefold()
        if scene_id in review or groups.setdefault(group, split) != split:
            raise ValueError(f"Duplicate review row or group leaking across splits: {group}")
        review[scene_id] = {**row, "group": group}
    exports = {"train": [], "valid": []}
    manifest = []
    seen_turns = {}
    found = set()
    for scene_id, _, scene in records(args.source):
        if scene_id not in review or scene_id in found:
            continue
        found.add(scene_id)
        row = review[scene_id]
        split = row["split"]
        source_turns = scene["turns"]
        dialogue_id = fingerprint(source_turns)
        if dialogue_id in seen_turns:
            if seen_turns[dialogue_id] != split:
                raise ValueError(f"Duplicate dialogue across train and valid: {scene_id}")
            continue
        seen_turns[dialogue_id] = split
        # The reviewer approves an exact contiguous window by visible turn index.
        start, end = row["start"], row["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source_turns):
            raise ValueError(f"Invalid start/end for {scene_id}")
        window = source_turns[start:end]
        if len(window) < 2 or window[-2]["role"] != "user" or window[-1]["role"] != "assistant":
            raise ValueError(f"{scene_id}: approved window must end in user/assistant turns")
        system = clean(row["system"])
        if not system:
            raise ValueError(f"{scene_id}: supply the reviewed, concise character card as system")
        exports[split].append({"messages": [{"role": "system", "content": system}, *window]})
        manifest.append({"scene_id": scene_id, "group": row["group"], "split": split,
                         "start": start, "end": end, "tags": row.get("tags", []),
                         "source": "PygmalionAI/PIPPA"})
    if set(review) != found:
        raise ValueError(f"Missing source rows: {sorted(set(review) - found)[:5]}")
    if not all(exports.values()):
        raise ValueError("Approve at least one window in each split.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for split, data in exports.items():
        write_rows(out / f"{split}.jsonl", data)
    write_rows(out / "manifest.jsonl", manifest)
    print({split: len(data) for split, data in exports.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index_cmd = commands.add_parser("index")
    index_cmd.add_argument("source")
    index_cmd.add_argument("output")
    index_cmd.add_argument("--pattern")
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("source")
    prepare_cmd.add_argument("review")
    prepare_cmd.add_argument("output")
    args = parser.parse_args()
    if args.command == "index":
        index(args)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
