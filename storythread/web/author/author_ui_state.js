const DRAFTS = "storythread-author-drafts-v2";
const NAMES = ["recent", "earlier"];
const TITLE = {recent: "Recent evidence", earlier: "Earlier evidence"};
const HINT = {
  recent: "The reply depends on something said a few turns ago.",
  earlier: "The reply depends on something said much further back.",
};
const CARD_HINT = "The system prompt both endings are trained with.";
const GROUP_HINT = "Scenes of one character or world share a group, so they " +
  "stay together in the train/valid split.";

function draftOk(draft) {
  return Boolean(draft.branches);
}

// Stores the open scene so a refresh can restore it.
function persist() {
  try {
    const data = readDrafts();
    data.items[draftKey()] = {
      originalId: state.originalId, scenario_id: state.scenario_id,
      group: state.group, system: state.system, history: state.history,
      branches: state.branches, dirty: true,
    };
    writeDrafts(data);
  } catch (error) {
    ui.serverError = "Could not keep unsaved edits in this browser.";
  }
}

// ---- Scene state --------------------------------------------------------

function blankBranch() {
  return {user: "", reply: "", evidence: "", source: null, missing: false};
}

// Returns a new scene with a generated id and a three-turn start.
function blankScene() {
  return {
    originalId: null, scenario_id: nextId(), group: "", system: "",
    history: ["assistant", "user", "assistant"].map((role) => (
      {key: newKey(), role: role, content: ""})),
    branches: {recent: blankBranch(), earlier: blankBranch()},
    dirty: false,
  };
}

// Returns one ending; evidence_turn becomes a turn key or "self".
function branchFrom(row, name, turns) {
  const raw = (row.branches && row.branches[name]) || {};
  const index = raw.evidence_turn;
  let source = null;
  let missing = false;
  if (index === turns.length) source = "self";
  else if (Number.isInteger(index) && turns[index]) source = turns[index].key;
  else if (Number.isInteger(index)) missing = true;
  return {
    user: raw.user || "", reply: raw.reply || "",
    evidence: typeof raw.evidence === "string" ? raw.evidence : "",
    source: source, missing: missing,
  };
}

// Returns editor state for a row loaded from the server.
function fromServer(row) {
  const turns = (row.history || []).map((turn) => ({
    key: newKey(), role: turn.role || "assistant",
    content: turn.content || "",
  }));
  return {
    originalId: row.scenario_id, scenario_id: row.scenario_id || "",
    group: row.group || "", system: row.system || "", history: turns,
    branches: {
      recent: branchFrom(row, "recent", turns),
      earlier: branchFrom(row, "earlier", turns),
    },
    dirty: false,
  };
}

// Returns the history index an ending cites, or null.
function evidenceIndex(branch) {
  if (!branch.source || branch.missing) return null;
  if (branch.source === "self") return state.history.length;
  const index = state.history.findIndex((t) => t.key === branch.source);
  return index < 0 ? null : index;
}

// Returns how many turns the cited turn sits from the final reply.
function turnsBack(branch) {
  const index = evidenceIndex(branch);
  return index === null ? null : state.history.length + 1 - index;
}

// Returns the cited turn's text, or the ending's own user turn.
function citedText(branch) {
  if (!branch.source || branch.missing) return "";
  if (branch.source === "self") return branch.user;
  const turn = state.history.find((t) => t.key === branch.source);
  return turn ? turn.content : "";
}

// Returns the scene in the authored file shape.
function payload() {
  const branch = (name) => {
    const item = state.branches[name];
    return {
      user: item.user, reply: item.reply,
      evidence_turn: evidenceIndex(item), evidence: item.evidence,
    };
  };
  return {
    scenario_id: state.scenario_id, group: state.group,
    system: state.system,
    history: state.history.map((t) => ({role: t.role, content: t.content})),
    branches: {recent: branch("recent"), earlier: branch("earlier")},
  };
}

// ---- Validation: only shown after Save is pressed -----------------------

// Returns [{text, target}] for everything wrong with the open scene.
function problems() {
  const out = [];
  const add = (text, target) => out.push({text: text, target: target});
  const turns = state.history;
  if (!state.scenario_id.trim()) add("scenario_id is empty.", "scene");
  if (!state.group.trim()) add("group is empty.", "scene");
  if (!state.system.trim()) add("Character card is empty.", "scene");
  if (!turns.length) add("The story needs at least one turn.", "scene");
  turns.forEach((turn, i) => {
    const target = "turn:" + turn.key;
    if (i > 0 && turn.role === turns[i - 1].role) {
      add(`Turn ${i} has the same speaker as turn ${i - 1}.`, target);
    }
    if (!turn.content.trim()) add(`Turn ${i} is empty.`, target);
  });
  const last = turns[turns.length - 1];
  if (last && last.role !== "assistant") {
    add("The story must end with a Character turn.", "turn:" + last.key);
  }
  const back = {};
  NAMES.forEach((name) => {
    const b = state.branches[name];
    const target = "ending:" + name;
    const label = TITLE[name];
    if (!b.user.trim()) add(`${label}: final user turn is empty.`, target);
    if (!b.reply.trim()) add(`${label}: ideal reply is empty.`, target);
    if (b.missing) add(`${label}: cited turn was removed.`, target);
    else if (!b.source) add(`${label}: choose an evidence turn.`, target);
    if (!b.evidence.trim()) {
      add(`${label}: evidence quote is empty.`, target);
    } else if (b.source && !b.missing &&
               !citedText(b).trim().includes(b.evidence)) {
      add(`${label}: evidence quote is not in the cited turn.`, target);
    }
    back[name] = turnsBack(b);
  });
  if (back.recent !== null && back.earlier !== null &&
      back.earlier <= back.recent) {
    add("Earlier evidence must be farther from the reply than recent " +
        "evidence.", "ending:earlier");
  }
  return out;
}
