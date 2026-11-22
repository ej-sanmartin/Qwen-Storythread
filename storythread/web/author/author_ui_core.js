// Helpers, drafts, and status shared by every authoring mode.

const $ = (id) => document.getElementById(id);
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
  const ok = draft && Array.isArray(draft.history) && draftOk(draft);
  return ok ? draft : null;
}

function dropDraft(key) {
  const data = readDrafts();
  delete data.items[key];
  writeDrafts(data);
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

// Returns the next unused scene-N id.
function nextId() {
  const used = new Set(scenes.map((item) => item.scenario_id));
  let number = scenes.length + 1;
  while (used.has("scene-" + number)) number += 1;
  return "scene-" + number;
}
