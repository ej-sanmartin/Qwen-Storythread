"""Validate authored training scene pairs and held-out evaluation scenarios.

Turn indices count conversation turns only; the system card is excluded.
Evidence distance is the index of the target reply minus the evidence index.
"""

BRANCHES = ("recent", "earlier")
VERSIONS = ("a", "b")


def text(value, where):
    """Returns a stripped nonempty string or raises ValueError."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected nonempty text")
    return value.strip()


def turns_of(items, where):
    """Returns validated, alternating {role, content} turns."""
    if not isinstance(items, list) or not items:
        raise ValueError(f"{where}: expected a nonempty list of turns")
    turns = []
    for number, item in enumerate(items):
        role = item.get("role") if isinstance(item, dict) else None
        if role not in ("user", "assistant"):
            raise ValueError(f"{where}[{number}]: role must be user or assistant")
        if turns and turns[-1]["role"] == role:
            raise ValueError(f"{where}[{number}]: roles must alternate")
        turns.append({"role": role, "content": text(item.get("content"), where)})
    return turns


def check_evidence(turns, index, quote, where):
    """Checks that the evidence index exists and its turn contains the quote."""
    if type(index) is not int or not 0 <= index < len(turns):
        raise ValueError(f"{where}: evidence_turn is outside the conversation")
    if quote is not None and quote not in turns[index]["content"]:
        raise ValueError(f"{where}: evidence quote not found in turn {index}")


def parse_authored(data):
    """Returns authored scene pairs with a validated recent and earlier branch.

    Each row has scenario_id, group, system, history (ends with an assistant
    turn), and branches {recent, earlier} of {user, reply, evidence_turn,
    evidence}. evidence_turn may point at the branch's own user prompt.
    """
    seen, result = set(), []
    for row in data:
        sid = text(row.get("scenario_id"), "scenario_id")
        where = f"scenario {sid}"
        if sid in seen:
            raise ValueError(f"Duplicate scenario_id: {sid}")
        seen.add(sid)
        history = turns_of(row.get("history"), f"{where} history")
        if history[-1]["role"] != "assistant":
            raise ValueError(f"{where}: history must end with an assistant turn")
        branches, distances = {}, {}
        for name in BRANCHES:
            raw = (row.get("branches") or {}).get(name)
            if not isinstance(raw, dict):
                raise ValueError(f"{where}: missing branch {name}")
            user, reply = text(raw.get("user"), where), text(raw.get("reply"), where)
            turns = history + [{"role": "user", "content": user},
                               {"role": "assistant", "content": reply}]
            index = raw.get("evidence_turn")
            check_evidence(turns[:-1], index, raw.get("evidence"), f"{where} {name}")
            distances[name] = len(history) + 1 - index
            branches[name] = {"user": user, "reply": reply, "evidence_turn": index,
                              "evidence": raw.get("evidence"),
                              "distance_turns": distances[name]}
        if distances["earlier"] <= distances["recent"]:
            raise ValueError(f"{where}: earlier evidence must be farther than recent")
        result.append({"scenario_id": sid, "group": text(row.get("group"), where).casefold(),
                       "system": text(row.get("system"), where),
                       "history": history, "branches": branches})
    if not result:
        raise ValueError("No authored scenarios")
    return result


def parse_scenarios(data):
    """Returns evaluation scenarios of two versions with an identical final prompt.

    Each row has scenario_id, group, and versions {a, b} of {messages,
    expected, evidence_turn, evidence}. messages are system plus turns ending
    in a user turn. The versions differ at the evidence turn.
    """
    seen, result = set(), []
    for row in data:
        sid = text(row.get("scenario_id"), "scenario_id")
        where = f"scenario {sid}"
        if sid in seen:
            raise ValueError(f"Duplicate scenario_id: {sid}")
        seen.add(sid)
        versions = row.get("versions") or {}
        if sorted(versions) != list(VERSIONS):
            raise ValueError(f"{where}: versions must be exactly a and b")
        parsed = {}
        for name in VERSIONS:
            raw = versions[name]
            messages = raw.get("messages")
            card = messages[0] if isinstance(messages, list) and messages else {}
            if card.get("role") != "system":
                raise ValueError(f"{where} {name}: messages must start with a system card")
            turns = turns_of(messages[1:], f"{where} {name}")
            if turns[-1]["role"] != "user":
                raise ValueError(f"{where} {name}: messages must end with a user turn")
            index = raw.get("evidence_turn")
            check_evidence(turns, index, text(raw.get("evidence"), where), f"{where} {name}")
            system = {"role": "system", "content": text(card.get("content"), where)}
            parsed[name] = {"messages": [system, *turns],
                            "expected": text(raw.get("expected"), where),
                            "evidence_turn": index, "evidence": raw["evidence"]}
        a, b = parsed["a"], parsed["b"]
        if a["messages"][0] != b["messages"][0] or a["messages"][-1] != b["messages"][-1]:
            raise ValueError(f"{where}: system card and final user prompt must match")
        if a["evidence_turn"] != b["evidence_turn"]:
            raise ValueError(f"{where}: both versions must cite the same evidence turn")
        if a["messages"][1 + a["evidence_turn"]] == b["messages"][1 + b["evidence_turn"]]:
            raise ValueError(f"{where}: versions must differ at the evidence turn")
        distance = len(a["messages"]) - 1 - a["evidence_turn"]
        result.append({"scenario_id": sid, "group": text(row.get("group"), where).casefold(),
                       "versions": parsed, "distance": distance})
    if not result:
        raise ValueError("No scenarios")
    return result


def check_disjoint(first, second):
    """Raises if two scenario lists share an ID, group, or character card."""
    def keys(scenarios):
        return ({s["scenario_id"] for s in scenarios}, {s["group"] for s in scenarios},
                {v["messages"][0]["content"] for s in scenarios for v in s["versions"].values()})
    for label, one, two in zip(("scenario_id", "group", "system card"), keys(first), keys(second)):
        if one & two:
            raise ValueError(f"Development and test sets share a {label}: {sorted(one & two)[:3]}")
