// ---- Sidebar (eval) -----------------------------------------------------

// Removes a turn. Removing the evidence turn clears the choice.
function removeTurn(key) {
  const index = state.history.findIndex((t) => t.key === key);
  if (index < 0) return;
  state.history.splice(index, 1);
  if (state.evidence === key) state.evidence = null;
  const near = state.history[Math.max(0, index - 1)];
  ui.selected = near ? "turn:" + near.key : "scene";
  touch();
  renderChat();
  renderFields();
}

// Sets the evidence turn (turn) or clears it (null).
function setEvidence(turn) {
  state.evidence = turn ? turn.key : null;
  if (turn && !turn.alt) turn.alt = turn.content;
  touch();
  renderChat();
  renderFields();
}

// Copies the words highlighted in a textarea into a version's quote.
function useHighlight(name, areaId, note) {
  const area = $(areaId);
  const quote = area.value.slice(area.selectionStart, area.selectionEnd).trim();
  if (!quote) {
    note.textContent = "Highlight some words in the box above first.";
    return;
  }
  note.textContent = "";
  state.quotes[name] = quote;
  $("quote-" + name).value = quote;
  touch();
  renderChat();
}

// Returns the sidebar controls for the evidence turn's two texts.
function evidenceFields(turn) {
  const quotes = VERSIONS.map((name) => {
    const note = el("p", {class: "hint"});
    const areaId = name === "a" ? "turn-text" : "turn-alt";
    return [
      field({label: `Quote (${TITLE[name].toLowerCase()})`, rows: 2,
             value: state.quotes[name], id: "quote-" + name,
             set: (v) => { state.quotes[name] = v; },
             hint: "Exact words from this version of the turn."}),
      el("div", {class: "btns"}, el("button", {
        type: "button", "data-keepsel": true, text: "Use highlighted text",
        onclick: () => useHighlight(name, areaId, note)})),
      note,
    ];
  });
  return [
    field({label: "Text (version B)", value: turn.alt, rows: 7,
           id: "turn-alt", set: (v) => { turn.alt = v; }}),
    el("p", {class: "hint",
             text: `${plural(turnsBack())} from this turn to the final reply.`}),
    ...quotes.flat(),
    el("div", {class: "btns"}, el("button", {
      type: "button", text: "Not the evidence turn",
      onclick: () => setEvidence(null)})),
  ];
}

function turnFields(i, turn) {
  const mine = turn.key === state.evidence;
  const role = el("select", {}, ...["assistant", "user"].map((r) => (
    el("option", {value: r, text: `${SPEAKER[r]} (${r})`}))));
  role.value = turn.role;
  role.addEventListener("change", () => {
    turn.role = role.value;
    touch();
    renderChat();
  });
  const add = (text, at) => el("button", {
    type: "button", text: text,
    onclick: () => insertTurn(at, opposite(turn.role))});
  return [
    el("h2", {text: `Turn ${i}`}),
    el("label", {}, el("span", {text: "Speaker"}), role),
    field({label: mine ? "Text (version A)" : "Text", value: turn.content,
           rows: 7, id: "turn-text", set: (v) => { turn.content = v; }}),
    el("div", {class: "btns"}, add("Add turn above", i),
      add("Add turn below", i + 1),
      el("button", {type: "button", text: "Remove turn",
                    onclick: () => removeTurn(turn.key)})),
    el("h3", {text: "Evidence"}),
    ...(mine ? evidenceFields(turn) : [
      el("p", {class: "hint",
               text: "Make this the one turn whose text differs between " +
                 "the two versions."}),
      el("div", {class: "btns"}, el("button", {
        type: "button", text: "Use as evidence turn",
        onclick: () => setEvidence(turn)}))]),
  ];
}

// Returns the final player turn's controls.
function finalFields() {
  return [
    el("h2", {text: "Final player turn"}),
    field({label: "Text", value: state.final, rows: 4,
           set: (v) => { state.final = v; },
           hint: "The same prompt in both versions. The right reply " +
             "changes with the evidence turn."}),
  ];
}

// Returns the expected-behavior controls for one version.
function expectedFields(name) {
  return [
    el("h2", {text: `Expected \u00b7 ${TITLE[name]}`}),
    field({label: "Expected behavior", value: state.expected[name], rows: 4,
           set: (v) => { state.expected[name] = v; },
           hint: "What a correct reply does. You judge replies against " +
             "this note."}),
  ];
}

// Redraws the sidebar for the current selection.
function renderFields() {
  const [kind, id] = ui.selected.split(":");
  let kids = null;
  if (kind === "turn") {
    const i = state.history.findIndex((t) => t.key === id);
    if (i >= 0) kids = turnFields(i, state.history[i]);
  } else if (kind === "final") {
    kids = finalFields();
  } else if (kind === "expected" && TITLE[id]) {
    kids = expectedFields(id);
  }
  if (!kids) {
    ui.selected = "scene";
    kids = sceneFields();
  }
  $("fields").replaceChildren(...kids);
  markSelected();
}
