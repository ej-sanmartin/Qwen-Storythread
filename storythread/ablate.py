"""Apply an ablation plan to a copy of an MLX-LM LoRA checkpoint.

The plan is the JSON the adapter heatmap exports. This writes a new adapter
directory and does not generate or score replies.
"""
import hashlib
import json
import struct
from pathlib import Path

SCHEMA = 1
WEIGHTS = "adapters.safetensors"
CONFIG = "adapter_config.json"


def checkpoint_record(path):
    """Returns {path, sha256} for one checkpoint file.

    path is relative to the current directory when the file is inside it,
    so a plan exported from the project root stays usable there.
    """
    full = Path(path).resolve()
    try:
        shown = full.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        shown = full.as_posix()
    digest = hashlib.sha256(full.read_bytes()).hexdigest()
    return {"path": shown, "sha256": digest}


def _read_header(path):
    """Returns (bytes, header_size, header) for a safetensors file."""
    data = Path(path).read_bytes()
    try:
        (size,) = struct.unpack("<Q", data[:8])
        header = json.loads(data[8:8 + size])
    except (struct.error, ValueError) as error:
        raise ValueError(f"{path}: not a readable safetensors file") from error
    return data, size, header


def load_plan(path):
    """Returns a checked plan with sorted, unique module paths.

    path: the exported JSON. Raises ValueError when the schema, arm,
    checkpoint, sha256, or module list is missing or empty.
    """
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: not a readable ablation plan") from error
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema must be {SCHEMA}")
    arm, checkpoint = plan.get("arm"), plan.get("checkpoint")
    digest, modules = plan.get("sha256"), plan.get("modules")
    if not isinstance(arm, str) or not arm:
        raise ValueError(f"{path}: arm is missing")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{path}: checkpoint is missing")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{path}: sha256 is missing")
    if (not isinstance(modules, list) or not modules
            or not all(isinstance(item, str) and item for item in modules)):
        raise ValueError(f"{path}: modules must be non-empty module paths")
    plan["modules"] = sorted(set(modules))
    return plan


def _lora_b_keys(header, modules):
    """Returns the .lora_b key for every module, or raises if one is absent."""
    missing = []
    keys = []
    for module in modules:
        key = f"{module}.lora_b"
        info = header.get(key)
        if not isinstance(info, dict) or "data_offsets" not in info:
            missing.append(key)
        else:
            keys.append(key)
    if missing:
        listed = ", ".join(missing)
        raise ValueError(
            "plan names modules whose lora_b is not in the checkpoint: "
            + listed)
    return keys


def _zero_payloads(data, header_size, header, keys):
    """Returns a copy of the safetensors bytes with those payloads set to 0.

    The header, shapes, and dtypes are unchanged. Zero bytes are a zero
    value for F32, F16, and BF16.
    """
    out = bytearray(data)
    body = 8 + header_size
    for key in keys:
        begin, end = header[key]["data_offsets"]
        if not (0 <= begin <= end <= len(out) - body):
            raise ValueError(f"{key}: offsets fall outside the file")
        out[body + begin:body + end] = b"\x00" * (end - begin)
    return bytes(out)


def apply_plan(plan_path, output):
    """Writes a new adapter directory with the plan's lora_b tensors zeroed.

    output must not already exist. The source checkpoint is only read.
    Returns the zeroed .lora_b keys. Raises ValueError on a bad plan, a
    hash mismatch, a missing module, or an existing output path.
    """
    output = Path(output)
    if output.exists():
        raise ValueError(f"{output} already exists")
    plan = load_plan(plan_path)
    source = Path(plan["checkpoint"])
    if not source.is_file():
        raise ValueError(f"{source} not found")
    data, size, header = _read_header(source)
    actual = hashlib.sha256(data).hexdigest()
    if actual != plan["sha256"]:
        raise ValueError(f"{source}: sha256 {actual} does not match the plan")
    keys = _lora_b_keys(header, plan["modules"])
    config = source.parent / CONFIG
    if not config.is_file():
        raise ValueError(f"{config} not found")
    weights = _zero_payloads(data, size, header, keys)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    (output / CONFIG).write_bytes(config.read_bytes())
    (output / WEIGHTS).write_bytes(weights)
    return keys
