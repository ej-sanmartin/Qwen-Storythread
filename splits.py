"""Assign train/valid splits so related examples never straddle them."""
from collections import Counter
import hashlib
import math


def digest(seed, key):
    """Returns a stable pseudo-random sort key."""
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def assign_splits(items, kinds, valid_share, seed):
    """Returns {item_id: "train" | "valid"}.

    items maps item_id to nodes (scenario, character group, bot ID, dialogue).
    Items that share any node are linked, and each linked component is assigned
    to a single split. kinds maps item_id to a source label (such as "pippa" or
    "authored"); each source gets about valid_share of its items in valid.
    """
    parent = {}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for nodes in items.values():
        for node in nodes:
            parent.setdefault(node, node)
        for node in nodes[1:]:
            parent[find(node)] = find(nodes[0])
    components = {}
    for item_id, nodes in items.items():
        components.setdefault(find(nodes[0]), []).append(item_id)
    quota = {kind: max(1, math.ceil(valid_share * count))
             for kind, count in Counter(kinds.values()).items()}
    filled = Counter()
    split = {}
    for ids in sorted(components.values(), key=lambda ids: digest(seed, min(ids))):
        counts = Counter(kinds[i] for i in ids)
        name = "valid" if any(filled[k] < quota[k] for k in counts) else "train"
        if name == "valid":
            filled.update(counts)
        split.update({item_id: name for item_id in ids})
    if set(split.values()) != {"train", "valid"}:
        raise ValueError("Need at least two unrelated groups to form both splits.")
    return split
