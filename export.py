"""Build the two training arms from a shared PIPPA core and authored pairs."""
import json
from pathlib import Path

from data import fingerprint, write_rows
from scenarios import BRANCHES
from splits import assign_splits, digest

SPLITS = ("train", "valid")


def authored_example(scene, branch):
    """Returns one chat example and manifest row for an authored branch."""
    data = scene["branches"][branch]
    messages = [{"role": "system", "content": scene["system"]}, *scene["history"],
                {"role": "user", "content": data["user"]},
                {"role": "assistant", "content": data["reply"]}]
    manifest = {"source": "authored", "scenario_id": scene["scenario_id"],
                "group": scene["group"], "branch": branch,
                "evidence_turn": data["evidence_turn"],
                "evidence_distance_turns": data["distance_turns"]}
    return {"messages": messages}, manifest


def build(core, authored, out, valid_share, authored_share, seed):
    """Writes out/<branch>/{train,valid,manifest}.jsonl and out/summary.json.

    core: reviewed PIPPA items {scene_id, nodes, example, manifest}.
    authored: output of scenarios.parse_authored.
    Both arms share the same core rows, row order, and authored scenarios.
    """
    items = {f"pippa:{c['scene_id']}": c["nodes"] for c in core}
    items.update({f"authored:{a['scenario_id']}":
                  [f"scenario:{a['scenario_id']}", f"group:{a['group']}"]
                  for a in authored})
    kinds = {item_id: item_id.split(":")[0] for item_id in items}
    splits = assign_splits(items, kinds, valid_share, seed)
    rows = {branch: {s: [] for s in SPLITS} for branch in BRANCHES}
    summary = {"seed": seed, "valid_share": valid_share,
               "authored_share_target": authored_share, "splits": {}}
    core_ids = []
    for split in SPLITS:
        scenes = [a for a in authored if splits[f"authored:{a['scenario_id']}"] == split]
        pool = sorted((c for c in core if splits[f"pippa:{c['scene_id']}"] == split),
                      key=lambda c: digest(seed, c["scene_id"]))
        if not scenes:
            raise ValueError(f"No authored scenarios landed in {split}; add more groups.")
        wanted = round(len(scenes) * (1 - authored_share) / authored_share)
        chosen = pool[:wanted]
        for item in chosen:
            for branch in BRANCHES:
                rows[branch][split].append((item["scene_id"], item["example"],
                                            {**item["manifest"], "split": split}))
            core_ids.append(fingerprint(item["example"]))
        for scene in scenes:
            for branch in BRANCHES:
                example, manifest = authored_example(scene, branch)
                rows[branch][split].append((scene["scenario_id"], example,
                                            {**manifest, "split": split}))
        total = len(chosen) + len(scenes)
        summary["splits"][split] = {
            "core": len(chosen), "core_available": len(pool), "authored": len(scenes),
            "authored_share": round(len(scenes) / total, 3)}
        if len(chosen) < wanted:
            summary["splits"][split]["warning"] = f"core short by {wanted - len(chosen)}"
    summary["core_sha"] = fingerprint(sorted(core_ids))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    summary["arms"] = {}
    for branch in BRANCHES:
        folder = out / branch
        folder.mkdir()
        manifest = []
        for split in SPLITS:
            ordered = sorted(rows[branch][split], key=lambda r: digest(seed, r[0]))
            write_rows(folder / f"{split}.jsonl", (r[1] for r in ordered))
            manifest += [{**r[2], "row": number} for number, r in enumerate(ordered)]
        write_rows(folder / "manifest.jsonl", manifest)
        summary["arms"][branch] = {s: len(rows[branch][s]) for s in SPLITS}
    if len({json.dumps(v, sort_keys=True) for v in summary["arms"].values()}) != 1:
        raise ValueError("Arms differ in example counts.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
