"""Compare how large two runs' saved LoRA updates are, as one HTML report.

Reads MLX-LM adapter snapshots directly (Qwen is never loaded) and writes a
standalone heatmap of the effective LoRA update RMS per block and projection.
This measures update size only. Behavioral contribution needs ablation.

The numbers work for any MLX-LM LoRA run, but the page's block types and
projection descriptions (inspect_adapters_glossary.js) are written for
Qwen3.5-4B. Other models plot without descriptions.
"""
import argparse
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

from .review_ui import assemble_page, page_paths

HTML = Path(__file__).with_name("inspect_adapters.html")
PAGE_STYLE = "inspect_adapters.css"
PAGE_SCRIPTS = ("inspect_adapters.js", "inspect_adapters_3d.js",
                "inspect_adapters_glossary.js")
METRIC = "Effective LoRA update RMS"
SNAPSHOT = re.compile(r"^(\d+)_adapters\.safetensors$")
FINAL = "adapters.safetensors"
FACTOR = re.compile(r"^(?P<module>.+)\.(?P<factor>lora_[ab])$")
BLOCK = re.compile(r"(?:^|\.)layers\.(?P<block>\d+)\.(?P<rest>.+)$")


def read_safetensors(path):
    """Returns {name: float32 array} for a .safetensors file.

    path: file written by MLX-LM. Supports F32, F16, and BF16 tensors; any
    other dtype or a damaged file raises ValueError.
    """
    data = Path(path).read_bytes()
    try:
        (size,) = struct.unpack("<Q", data[:8])
        header = json.loads(data[8:8 + size])
    except (struct.error, ValueError) as error:
        raise ValueError(f"{path}: not a readable safetensors file") from error
    body = memoryview(data)[8 + size:]
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        kind = info["dtype"]
        try:
            if kind == "BF16":
                bits = np.frombuffer(body[begin:end], dtype=np.uint16)
                array = (bits.astype(np.uint32) << 16).view(np.float32)
            elif kind in ("F32", "F16"):
                dtype = np.float32 if kind == "F32" else np.float16
                array = np.frombuffer(body[begin:end], dtype=dtype)
            else:
                raise ValueError(f"{path}: {name} has unsupported dtype {kind}")
            tensors[name] = array.astype(np.float32).reshape(info["shape"])
        except ValueError as error:
            if str(error).startswith(str(path)):
                raise
            raise ValueError(f"{path}: {name} does not match its shape") from error
    return tensors


def effective_rms(lora_a, lora_b, scale):
    """Returns RMS of the update scale * (A @ B) without forming A @ B.

    lora_a: (input, rank) and lora_b: (rank, output), as MLX-LM stores them.
    Uses ||A @ B||_F^2 = sum((A^T A) * (B B^T)), so only rank x rank
    matrices are built.
    """
    a, b = lora_a.astype(np.float64), lora_b.astype(np.float64)
    squared = float(np.sum((a.T @ a) * (b @ b.T)))
    return abs(scale) * math.sqrt(max(squared, 0.0) / (a.shape[0] * b.shape[1]))


def measure(tensors, scale, source):
    """Returns {module: {rms, in, out, rank}} from one snapshot's tensors.

    source: file name for error messages. Raises ValueError for a tensor
    that is not lora_a/lora_b, an unpaired factor, or incompatible shapes.
    """
    pairs = {}
    for name, array in tensors.items():
        found = FACTOR.match(name)
        if not found:
            raise ValueError(f"{source}: {name} is not a lora_a or lora_b tensor")
        pairs.setdefault(found["module"], {})[found["factor"]] = array
    stats = {}
    for module, factors in sorted(pairs.items()):
        if len(factors) != 2:
            have, = factors
            other = "lora_b" if have == "lora_a" else "lora_a"
            raise ValueError(f"{source}: {module} has {have} but no {other}")
        a, b = factors["lora_a"], factors["lora_b"]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"{source}: {module} is not an ordinary linear LoRA "
                             f"projection (shapes {a.shape}, {b.shape})")
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"{source}: {module} lora_a {a.shape} and lora_b {b.shape} do "
                "not share a rank (expected (input, rank) and (rank, output))")
        stats[module] = {"rms": effective_rms(a, b, scale), "in": a.shape[0],
                         "out": b.shape[1], "rank": a.shape[1]}
    return stats


def read_scale(folder):
    """Returns the LoRA scale from folder/adapter_config.json."""
    path = Path(folder) / "adapter_config.json"
    if not path.is_file():
        raise ValueError(f"{path} not found")
    config = json.loads(path.read_text(encoding="utf-8"))
    kind = config.get("fine_tune_type", "lora")
    if kind != "lora":
        raise ValueError(f"{path}: fine_tune_type {kind!r} is not supported")
    try:
        return float(config["lora_parameters"]["scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: lora_parameters.scale is missing") from error


def recorded_step(folder):
    """Returns last_iter from train_metrics.json, or None."""
    path = Path(folder) / "train_metrics.json"
    try:
        step = json.loads(path.read_text(encoding="utf-8")).get("last_iter")
    except (OSError, ValueError):
        return None
    return step if isinstance(step, int) else None


def find_checkpoints(folder):
    """Returns ({step: path}, notes) for one run directory.

    Numbered snapshots are NNNNNNN_adapters.safetensors. The final
    adapters.safetensors is listed once: dropped when its bytes equal the
    latest snapshot, otherwise placed at the step in train_metrics.json
    (skipped with a note when that step is unknown or already taken).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory")
    steps = {}
    for path in folder.iterdir():
        found = SNAPSHOT.match(path.name)
        if found:
            steps[int(found[1])] = path
    notes = []
    final = folder / FINAL
    if final.is_file() and not (steps and final.read_bytes() ==
                                steps[max(steps)].read_bytes()):
        step = recorded_step(folder)
        if step is None or step in steps:
            notes.append(f"{folder}: {FINAL} differs from the latest snapshot "
                         "and has no usable recorded step, so it was skipped.")
        else:
            steps[step] = final
    if not steps:
        raise ValueError(f"{folder}: no *_adapters.safetensors snapshots found")
    return dict(sorted(steps.items())), notes


def column_labels(columns):
    """Maps each column id to its last path part, or the full id if shared."""
    short = {column: column.rsplit(".", 1)[-1] for column in columns}
    if len(set(short.values())) < len(short):
        return {column: column for column in columns}
    return short


def build_report(runs):
    """Returns the JSON-ready report for {label: run directory}."""
    found, scales, notes = {}, {}, []
    for label, folder in runs.items():
        scales[label] = read_scale(folder)
        found[label], extra = find_checkpoints(folder)
        notes += extra
    steps = sorted(set().union(*found.values()))
    rms = {label: {} for label in runs}
    dims = {label: {} for label in runs}
    for label, snapshots in found.items():
        for step, path in snapshots.items():
            stats = measure(read_safetensors(path), scales[label], path)
            rms[label][str(step)] = {m: float(f"{s['rms']:.6g}")
                                     for m, s in stats.items()}
            for module, s in stats.items():
                shape = {"in": s["in"], "out": s["out"], "rank": s["rank"]}
                if dims[label].setdefault(module, shape) != shape:
                    raise ValueError(f"{path}: {module} changed shape")
    paths = sorted({m for per_label in dims.values() for m in per_label})
    parsed = {}
    for module in paths:
        found_block = BLOCK.search(module)
        parsed[module] = ((int(found_block["block"]), found_block["rest"])
                          if found_block else (None, module))
    columns = sorted({column for _, column in parsed.values()})
    labels = column_labels(columns)
    blocks = sorted({block for block, _ in parsed.values()},
                    key=lambda block: (block is None, block or 0))
    values = [v for per in rms.values() for snap in per.values()
              for v in snap.values()]
    return {
        "metric": METRIC, "arms": list(runs), "scales": scales,
        "steps": steps, "notes": notes,
        "missing": {label: [s for s in steps if s not in found[label]]
                    for label in runs},
        "blocks": blocks,
        "columns": [{"id": column, "label": labels[column]}
                    for column in columns],
        "modules": {module: {"block": parsed[module][0],
                             "column": parsed[module][1],
                             "dims": {label: dims[label][module]
                                      for label in runs
                                      if module in dims[label]}}
                    for module in paths},
        "rms": rms, "max": max(values, default=0.0),
    }


def render(report):
    """Returns the standalone HTML page with the report embedded."""
    page = assemble_page(HTML, PAGE_STYLE, PAGE_SCRIPTS).decode("utf-8")
    data = json.dumps(report, separators=(",", ":")).replace("<", "\\u003c")
    return page.replace("{{data}}", data, 1).encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent", required=True, help="recent run directory")
    parser.add_argument("--earlier", required=True, help="earlier run directory")
    parser.add_argument("--output", default="work/adapter_heatmap.html")
    args = parser.parse_args(argv)
    missing = [p for p in page_paths(HTML, PAGE_STYLE, PAGE_SCRIPTS)
               if not p.is_file()]
    if missing:
        raise SystemExit("Missing page files: " + ", ".join(map(str, missing)))
    try:
        report = build_report({"recent": args.recent, "earlier": args.earlier})
    except ValueError as error:
        raise SystemExit(f"error: {error}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(render(report))
    for note in report["notes"]:
        print(f"note: {note}")
    for label, steps in report["missing"].items():
        if steps:
            print(f"note: {label} has no snapshot at step(s) {steps}")
    print(f"Wrote {output}: {len(report['steps'])} step(s), "
          f"{len(report['modules'])} module(s)")


if __name__ == "__main__":
    main()
