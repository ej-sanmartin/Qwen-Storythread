"""Index PIPPA, review windows, and export two matched training datasets."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import re

TAGS = {
    "fantasy": re.compile(r"\b(fantasy|magic\w*|mage|wizard|witch\w*|sorcer\w*|dragon|"
                          r"elf|elves|knight|kingdom|castle|quest|spell)\b", re.I),
    "romance": re.compile(r"\b(romance|romantic|lover|kiss\w*|flirt\w*|"
                          r"affection|crush|courtship)\b", re.I),
}
PREVIEW_CHARS = 1200


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
    """Yields (scene_id, default group, scene) for each usable PIPPA row.

    The default group is the PIPPA bot_id, or the normalized bot name if absent.
    """
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
        bot_id = clean(row.get("bot_id"))
        group = bot_id or " ".join(re.findall(r"\w+", name.casefold()))
        scene = {"name": name, "description": description, "turns": turns}
        yield fingerprint(scene), group, {**scene, "bot_id": bot_id}


def index(args):
    """Writes every distinct PIPPA scene; --pattern and --tag only narrow it."""
    pattern = re.compile(args.pattern, re.IGNORECASE) if args.pattern else None
    seen = set()

    def candidates():
        for scene_id, group, scene in records(args.source):
            if scene_id in seen:
                continue
            text = " ".join([scene["name"], scene["description"],
                             *(t["content"] for t in scene["turns"])])
            if pattern and not pattern.search(text):
                continue
            tags = [tag for tag in args.tag or [] if TAGS[tag].search(text)]
            if args.tag and not tags:
                continue
            seen.add(scene_id)
            yield {"scene_id": scene_id, "suggested_group": group,
                   "tags": tags, **scene}

    write_rows(args.output, candidates())


def review(args):
    """Writes a review template with a proposed final window per scene."""
    def template():
        picked = heapq.nsmallest(
            args.sample, rows(args.index),
            key=lambda r: hashlib.sha256(
                f"{args.seed}:{r['scene_id']}".encode()).hexdigest())
        for scene in picked:
            turns = scene["turns"]
            end = len(turns) if turns[-1]["role"] == "assistant" else len(turns) - 1
            if end < 2:
                continue
            yield {"scene_id": scene["scene_id"], "approved": False,
                   "group": scene["suggested_group"], "bot_id": scene["bot_id"],
                   "tags": scene["tags"], "start": max(0, end - args.context),
                   "end": end, "system": scene["description"][:PREVIEW_CHARS]}

    write_rows(args.output, template())


def show(args):
    """Prints one indexed scene with turn numbers for window selection."""
    for scene in rows(args.index):
        if scene["scene_id"] == args.scene_id:
            print(f"{scene['name']} (group {scene['suggested_group']})\n{scene['description']}\n")
            for number, turn in enumerate(scene["turns"]):
                print(f"[{number}] {turn['role']}: {turn['content']}\n")
            return
    raise SystemExit(f"Scene not found: {args.scene_id}")


def prepare(args):
    """Exports recent/ and earlier/ datasets from approved review rows."""
    from export import build
    from scenarios import parse_authored

    review = {}
    for row in rows(args.review):
        if type(row.get("approved")) is not bool:
            raise ValueError("Each review row needs approved: true or false.")
        if row["scene_id"] in review:
            raise ValueError(f"Duplicate review row: {row['scene_id']}")
        review[row["scene_id"]] = row
    approved = {k: v for k, v in review.items() if v["approved"]}
    core, found = [], set()
    for scene_id, _, scene in records(args.source):
        if scene_id not in approved or scene_id in found:
            continue
        found.add(scene_id)
        row, turns = approved[scene_id], scene["turns"]
        start, end = row["start"], row["end"]
        # The reviewer approves an exact contiguous window by visible turn index.
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(turns):
            raise ValueError(f"Invalid start/end for {scene_id}")
        window = turns[start:end]
        if len(window) < 2 or window[-2]["role"] != "user" or window[-1]["role"] != "assistant":
            raise ValueError(f"{scene_id}: approved window must end in user/assistant turns")
        system, group = clean(row.get("system")), clean(row.get("group")).casefold()
        if not system or not group:
            raise ValueError(f"{scene_id}: supply a reviewed card as system and a group")
        # Link by reviewer group, PIPPA bot, and window text so none straddle splits.
        nodes = [f"group:{group}", f"dialogue:{fingerprint(window)}"]
        if scene["bot_id"]:
            nodes.append(f"bot:{scene['bot_id']}")
        core.append({"scene_id": scene_id, "nodes": nodes,
                     "example": {"messages": [{"role": "system", "content": system}, *window]},
                     "manifest": {"source": "PygmalionAI/PIPPA", "scene_id": scene_id,
                                  "group": group, "bot_id": scene["bot_id"],
                                  "start": start, "end": end, "tags": row.get("tags", [])}})
    if set(approved) != found:
        raise ValueError(f"Missing source rows: {sorted(set(approved) - found)[:5]}")
    if not 0 < args.authored_share < 1 or not 0 < args.valid_share < 1:
        raise ValueError("Shares must be between 0 and 1.")
    build(core, parse_authored(rows(args.authored)), args.output,
          args.valid_share, args.authored_share, args.seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index_cmd = commands.add_parser("index")
    index_cmd.add_argument("source")
    index_cmd.add_argument("output")
    index_cmd.add_argument("--pattern", help="optional regex search aid")
    index_cmd.add_argument("--tag", action="append", choices=sorted(TAGS))
    review_cmd = commands.add_parser("review")
    review_cmd.add_argument("index")
    review_cmd.add_argument("output")
    review_cmd.add_argument("--sample", type=int, default=200)
    review_cmd.add_argument("--context", type=int, default=12)
    review_cmd.add_argument("--seed", type=int, default=42)
    show_cmd = commands.add_parser("show")
    show_cmd.add_argument("index")
    show_cmd.add_argument("scene_id")
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("source")
    prepare_cmd.add_argument("review")
    prepare_cmd.add_argument("authored")
    prepare_cmd.add_argument("output")
    prepare_cmd.add_argument("--valid-share", type=float, default=0.1)
    prepare_cmd.add_argument("--authored-share", type=float, default=0.25)
    prepare_cmd.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    {"index": index, "review": review, "show": show, "prepare": prepare}[args.command](args)


if __name__ == "__main__":
    main()
