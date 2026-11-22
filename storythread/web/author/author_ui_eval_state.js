const DRAFTS = "storythread-eval-drafts-v1";
const VERSIONS = ["a", "b"];
const TITLE = {a: "Version A", b: "Version B"};
const CARD_HINT = "The system prompt both versions are run with.";
const GROUP_HINT = "Scenarios of one character share a group. Dev and " +
  "test never share a group or a card.";

// ---- Eval scenario state ------------------------------------------------
// One shared history. The evidence turn has two texts: content (version A)
// and alt (version B). The final player turn is the same in both versions.

// Rejects drafts saved in another shape.
function draftOk(draft) {
  return Boolean(draft.quotes && draft.expected);
}

// Stores the open scenario so a refresh can restore it.
function persist() {
  try {
    const data = readDrafts();
    data.items[draftKey()] = {
      originalId: state.originalId, scenario_id: state.scenario_id,
      group: state.group, system: state.system, history: state.history,
      evidence: state.evidence, quotes: state.quotes, final: state.final,
      expected: state.expected, dirty: true,
    };
    writeDrafts(data);
  } catch (error) {
    ui.serverError = "Could not keep unsaved edits in this browser.";
  }
}

// Returns a new scenario with a generated id and a three-turn start.
function blankScene() {
  return {
    originalId: null, scenario_id: nextId(), group: "", system: "",
    history: ["assistant", "user", "assistant"].map((role) => (
      {key: newKey(), role: role, content: "", alt: ""})),
    evidence: null, quotes: {a: "", b: ""}, final: "",
    expected: {a: "", b: ""}, dirty: false,
  };
}

// Returns editor state for a row loaded from the server.
function fromServer(row) {
  const a = row.versions.a;
  const b = row.versions.b;
  const turns = a.messages.slice(1, -1).map((turn, i) => ({
    key: newKey(), role: turn.role, content: turn.content,
    alt: b.messages[1 + i].content,
  }));
  const index = a.evidence_turn;
  return {
    originalId: row.scenario_id, scenario_id: row.scenario_id || "",
    group: row.group || "", system: a.messages[0].content, history: turns,
    evidence: Number.isInteger(index) && turns[index]
      ? turns[index].key : null,
    quotes: {a: a.evidence || "", b: b.evidence || ""},
    final: a.messages[a.messages.length - 1].content,
    expected: {a: a.expected || "", b: b.expected || ""}, dirty: false,
  };
}

// Returns the history index of the evidence turn, or null.
function evidenceIndex() {
  const index = state.history.findIndex((t) => t.key === state.evidence);
  return index < 0 ? null : index;
}

// Returns how many turns the evidence turn sits from the final reply.
function turnsBack() {
  const index = evidenceIndex();
  return index === null ? null : state.history.length + 1 - index;
}

// Returns the scenario in the eval file shape.
function payload() {
  const version = (name) => ({
    messages: [
      {role: "system", content: state.system},
      ...state.history.map((turn) => ({
        role: turn.role,
        content: name === "b" && turn.key === state.evidence
          ? turn.alt : turn.content,
      })),
      {role: "user", content: state.final},
    ],
    expected: state.expected[name], evidence_turn: evidenceIndex(),
    evidence: state.quotes[name],
  });
  return {
    scenario_id: state.scenario_id, group: state.group,
    versions: {a: version("a"), b: version("b")},
  };
}

// Copies the server's stripped text back, keeping turn keys.
function applySaved(scene) {
  const a = scene.versions.a;
  const b = scene.versions.b;
  state.scenario_id = scene.scenario_id;
  state.group = scene.group;
  state.system = a.messages[0].content;
  state.history.forEach((turn, i) => {
    turn.role = a.messages[1 + i].role;
    turn.content = a.messages[1 + i].content;
    turn.alt = b.messages[1 + i].content;
  });
  state.final = a.messages[a.messages.length - 1].content;
  state.quotes = {a: a.evidence || "", b: b.evidence || ""};
  state.expected = {a: a.expected || "", b: b.expected || ""};
}

// ---- Validation: only shown after Save is pressed -----------------------

// Returns [{text, target}] for everything wrong with the open scenario.
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
    add("The story must end with a Character turn, so the final Player " +
        "turn can follow.", "turn:" + last.key);
  }
  if (!state.final.trim()) add("Final player turn is empty.", "final");
  const turn = turns.find((t) => t.key === state.evidence);
  if (!turn) {
    add("Choose the evidence turn: the one that differs.", "scene");
  } else {
    const target = "turn:" + turn.key;
    if (!turn.alt.trim()) add("Version B text is empty.", target);
    else if (turn.alt.trim() === turn.content.trim()) {
      add("Versions A and B must differ at the evidence turn.", target);
    }
    VERSIONS.forEach((name) => {
      const quote = state.quotes[name].trim();
      const text = name === "a" ? turn.content : turn.alt;
      if (!quote) add(`${TITLE[name]}: evidence quote is empty.`, target);
      else if (!text.includes(quote)) {
        add(`${TITLE[name]}: evidence quote is not in its turn text.`,
            target);
      }
    });
  }
  VERSIONS.forEach((name) => {
    if (!state.expected[name].trim()) {
      add(`${TITLE[name]}: expected behavior is empty.`, "expected:" + name);
    }
  });
  return out;
}
