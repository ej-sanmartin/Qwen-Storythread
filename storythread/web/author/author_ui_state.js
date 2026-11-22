const $ = (id) => document.getElementById(id);
const DRAFTS = "storythread-author-drafts-v2";
const NAMES = ["recent", "earlier"];
const TITLE = {recent: "Recent evidence", earlier: "Earlier evidence"};
const HINT = {
  recent: "The reply depends on something said a few turns ago.",
  earlier: "The reply depends on something said much further back.",
};
const SPEAKER = {assistant: "Character", user: "Player"};
const HEADERS = {"Content-Type": "application/json"};
let scenes = [];
let state = null;
let ui = {selected: "scene", showErrors: false, serverError: "", note: ""};

// Builds an element. attrs: class, text, onX handlers, or plain attributes.
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, val]) => {
    if (val == null || val === false) return;
    if (key === "class") node.className = val;
    else if (key === "text") node.textContent = val;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), val);
    else node.setAttribute(key, val === true ? "" : val);
  });
  node.append(...kids.filter((kid) => kid != null && kid !== false));
  return node;
}

// Returns "1 turn" or "N turns".
function plural(count) {
  return count + (count === 1 ? " turn" : " turns");
}

// Returns a short single-line preview of turn text.
function snippet(text) {
  const flat = text.trim().replace(/\s+/g, " ");
  if (!flat) return "(empty)";
  return flat.length > 28 ? flat.slice(0, 28) + "\u2026" : flat;
}

// Returns the other speaker.
function opposite(role) {
  return role === "assistant" ? "user" : "assistant";
}

// Returns a stable id for a history turn, so citations follow the turn.
function newKey() {
  return "t" + Math.random().toString(36).slice(2, 9);
}

// Returns parsed JSON or throws the server's error text.
async function readJson(response) {
  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error("The editor returned an unreadable response.");
  }
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
}

// ---- Drafts: unfinished work survives a refresh -------------------------

function readDrafts() {
  try {
    const data = JSON.parse(localStorage.getItem(DRAFTS) || "");
    if (data && data.items) return data;
  } catch (error) {
    // A damaged draft store is ignored.
  }
  return {items: {}};
}

function writeDrafts(data) {
  localStorage.setItem(DRAFTS, JSON.stringify(data));
}

// Returns the draft slot for the open scene.
function draftKey() {
  return state.originalId || "new";
}

// Returns the usable draft for key, or null.
function loadDraft(key) {
  const draft = readDrafts().items[key];
  const ok = draft && Array.isArray(draft.history) && draft.branches;
  return ok ? draft : null;
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

function dropDraft(key) {
  const data = readDrafts();
  delete data.items[key];
  writeDrafts(data);
}

// ---- Scene state --------------------------------------------------------

function blankBranch() {
  return {user: "", reply: "", evidence: "", source: null, missing: false};
}

// Returns the next unused scene-N id.
function nextId() {
  const used = new Set(scenes.map((item) => item.scenario_id));
  let number = scenes.length + 1;
  while (used.has("scene-" + number)) number += 1;
  return "scene-" + number;
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

// Draws the problems box. Empty until Save has been pressed once.
function renderProblems() {
  const list = ui.showErrors ? problems() : [];
  if (ui.serverError) list.unshift({text: ui.serverError, target: null});
  const box = $("problems");
  box.hidden = !list.length;
  box.replaceChildren();
  if (list.length) {
    box.append(el("h3", {text: "Problems"}), el("ul", {}, ...list.map((p) => (
      el("li", {}, p.target ? el("button", {
        type: "button", text: p.text, onclick: () => select(p.target),
      }) : p.text)))));
  }
  markBad();
}

function renderStatus() {
  const note = $("state");
  note.textContent = ui.note || (state.dirty ? "Unsaved changes" : "");
  note.className = ui.note.startsWith("Saved") ? "ok" : "hint";
  $("title").textContent = state.scenario_id || "New scene";
  document.title = (state.scenario_id || "New scene") +
    " \u00b7 Storythread Author";
}

// Marks the scene changed, stores a draft, and refreshes the chrome.
function touch() {
  state.dirty = true;
  ui.note = "";
  ui.serverError = "";
  persist();
  renderStatus();
  renderProblems();
  $("tokens").replaceChildren();
}
