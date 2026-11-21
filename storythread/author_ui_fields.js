// ---- Sidebar ------------------------------------------------------------

// Returns a labeled input (rows = 0) or textarea bound to set().
function field({label, value, set, hint, rows = 0, id}) {
  const control = rows
    ? el("textarea", {rows: rows, id: id})
    : el("input", {type: "text", id: id, autocomplete: "off"});
  control.value = value;
  control.addEventListener("input", () => {
    set(control.value);
    touch();
    renderChat();
  });
  return el("label", {}, el("span", {text: label}), control,
            hint ? el("span", {class: "hint", text: hint}) : null);
}

function sceneFields() {
  return [
    el("h2", {text: "Scene"}),
    field({label: "scenario_id", value: state.scenario_id,
           set: (v) => { state.scenario_id = v; },
           hint: "A unique label for this scene. Rename it if you like."}),
    field({label: "group", value: state.group,
           set: (v) => { state.group = v; },
           hint: "Scenes of one character or world share a group, so they " +
             "stay together in the train/valid split."}),
    field({label: "Character card", value: state.system, rows: 6,
           set: (v) => { state.system = v; },
           hint: "The system prompt both endings are trained with."}),
  ];
}

// Returns the current highlighted text for the selected bubble or textarea.
function selectedText() {
  const area = $("turn-text");
  if (area && document.activeElement === area &&
      area.selectionStart !== area.selectionEnd) {
    return area.value.slice(area.selectionStart, area.selectionEnd);
  }
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount || sel.isCollapsed) return "";
  const node = sel.getRangeAt(0).commonAncestorContainer;
  const host = (node.nodeType === 1 ? node : node.parentElement)
    .closest("[data-target]");
  return host && host.dataset.target === ui.selected ? sel.toString() : "";
}

// Makes the highlighted words of turn the evidence for an ending.
function useAsEvidence(name, turn, note) {
  const quote = selectedText().trim();
  if (!quote) {
    note.textContent = "Highlight some words in this turn first.";
    return;
  }
  const b = state.branches[name];
  b.source = turn.key;
  b.missing = false;
  b.evidence = quote;
  touch();
  renderChat();
  renderFields();
}

function insertTurn(index, role) {
  const turn = {key: newKey(), role: role, content: ""};
  state.history.splice(index, 0, turn);
  ui.selected = "turn:" + turn.key;
  touch();
  renderChat();
  renderFields();
}

function addTurn() {
  const last = state.history[state.history.length - 1];
  insertTurn(state.history.length, last ? opposite(last.role) : "assistant");
}

// Removes a turn. Endings that cited it keep their quote but are flagged.
function removeTurn(key) {
  const index = state.history.findIndex((t) => t.key === key);
  if (index < 0) return;
  state.history.splice(index, 1);
  NAMES.forEach((name) => {
    const b = state.branches[name];
    if (b.source === key) {
      b.source = null;
      b.missing = true;
    }
  });
  const near = state.history[Math.max(0, index - 1)];
  ui.selected = near ? "turn:" + near.key : "scene";
  touch();
  renderChat();
  renderFields();
}

function turnFields(i, turn) {
  const note = el("p", {class: "hint"});
  const role = el("select", {}, ...["assistant", "user"].map((r) => (
    el("option", {value: r, text: `${SPEAKER[r]} (${r})`}))));
  role.value = turn.role;
  role.addEventListener("change", () => {
    turn.role = role.value;
    touch();
    renderChat();
  });
  const cited = NAMES.filter((n) => state.branches[n].source === turn.key)
    .map((n) => el("p", {
      class: "hint",
      text: `${TITLE[n]}: \u201c${state.branches[n].evidence}\u201d \u00b7 ` +
        `${plural(turnsBack(state.branches[n]))} back`,
    }));
  const use = (name) => el("button", {
    type: "button", "data-keepsel": true, text: `Use as ${name} evidence`,
    onclick: () => useAsEvidence(name, turn, note),
  });
  return [
    el("h2", {text: `Turn ${i}`}),
    el("label", {}, el("span", {text: "Speaker"}), role),
    field({label: "Text", value: turn.content, rows: 7, id: "turn-text",
           set: (v) => { turn.content = v; }}),
    el("div", {class: "btns"},
      el("button", {type: "button", text: "Add turn above",
                    onclick: () => insertTurn(i, opposite(turn.role))}),
      el("button", {type: "button", text: "Add turn below",
                    onclick: () => insertTurn(i + 1, opposite(turn.role))}),
      el("button", {type: "button", text: "Remove turn",
                    onclick: () => removeTurn(turn.key)})),
    el("h3", {text: "Evidence"}),
    el("p", {class: "hint",
             text: "Highlight the words a later reply depends on, in the " +
               "chat or in the box above, then choose the ending."}),
    el("div", {class: "btns"}, use("recent"), use("earlier")),
    note, ...cited,
  ];
}

// Refills an ending's evidence dropdown from the current history.
function fillSources(select, b) {
  select.replaceChildren(
    el("option", {value: "", text: "Choose a turn"}),
    ...state.history.map((t, i) => el("option", {
      value: t.key,
      text: `${i} \u00b7 ${SPEAKER[t.role]} \u00b7 ${snippet(t.content)}`,
    })),
    el("option", {value: "self",
                  text: `${state.history.length} \u00b7 this ending's user turn`}));
  select.value = b.missing || !b.source ? "" : b.source;
}

// Updates an ending's cited text, distance, and removed-turn flag.
function refreshEvidence(name) {
  const b = state.branches[name];
  const first = document.querySelector("#ev-select option");
  first.textContent = b.missing ? "Cited turn was removed" : "Choose a turn";
  $("ev-preview").textContent = b.missing ? "" : citedText(b);
  const back = turnsBack(b);
  const line = $("ev-distance");
  line.className = "hint" + (b.missing ? " bad" : "");
  if (b.missing) line.textContent = "Cited turn was removed.";
  else if (back === null) line.textContent = "Choose the turn this reply uses.";
  else line.textContent = `${plural(back)} from this turn to the final reply.`;
}

// Returns the highlighted text inside node, or "".
function selectedIn(node) {
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount || sel.isCollapsed) return "";
  return node.contains(sel.getRangeAt(0).commonAncestorContainer)
    ? sel.toString() : "";
}

function endingFields(name) {
  const b = state.branches[name];
  const picker = el("select", {id: "ev-select"});
  fillSources(picker, b);
  picker.addEventListener("change", () => {
    b.source = picker.value || null;
    b.missing = false;
    touch();
    refreshEvidence(name);
    renderChat();
  });
  const preview = el("p", {class: "preview", id: "ev-preview"});
  const note = el("p", {class: "hint"});
  const copy = el("button", {
    type: "button", "data-keepsel": true, text: "Copy selection to quote",
    onclick: () => {
      const quote = selectedIn(preview).trim();
      if (!quote) {
        note.textContent = "Highlight words in the cited text first.";
        return;
      }
      note.textContent = "";
      b.evidence = quote;
      $("ev-quote").value = quote;
      touch();
      renderChat();
    },
  });
  return [
    el("h2", {text: TITLE[name]}),
    el("p", {class: "hint", text: HINT[name]}),
    field({label: "Final user turn", value: b.user, rows: 3,
           set: (v) => { b.user = v; refreshEvidence(name); }}),
    field({label: "Ideal character reply", value: b.reply, rows: 5,
           set: (v) => { b.reply = v; }}),
    el("h3", {text: "Evidence"}),
    el("label", {}, el("span", {text: "Evidence turn"}), picker),
    preview,
    el("div", {class: "btns"}, copy), note,
    el("p", {class: "hint", id: "ev-distance"}),
    field({label: "Evidence quote", value: b.evidence, rows: 2,
           id: "ev-quote",
           set: (v) => { b.evidence = v; },
           hint: "Exact words from the cited turn."}),
  ];
}

// Redraws the sidebar for the current selection.
function renderFields() {
  const [kind, id] = ui.selected.split(":");
  let kids = null;
  if (kind === "turn") {
    const i = state.history.findIndex((t) => t.key === id);
    if (i >= 0) kids = turnFields(i, state.history[i]);
  } else if (kind === "ending" && state.branches[id]) {
    kids = endingFields(id);
  }
  if (!kids) {
    ui.selected = "scene";
    kids = sceneFields();
  }
  $("fields").replaceChildren(...kids);
  if (kind === "ending" && state.branches[id]) refreshEvidence(id);
  markSelected();
}
