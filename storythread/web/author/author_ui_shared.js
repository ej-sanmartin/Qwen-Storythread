// Chat and sidebar pieces shared by every authoring mode.

// Writes text into p, underlining each cited quote.
function richText(p, text, cites) {
  const spans = [];
  cites.forEach((cite) => {
    const at = cite.quote ? text.indexOf(cite.quote) : -1;
    if (at >= 0) spans.push({at: at, end: at + cite.quote.length, name: cite.name});
  });
  spans.sort((a, b) => a.at - b.at);
  let pos = 0;
  spans.forEach((span) => {
    if (span.at < pos) return;
    p.append(text.slice(pos, span.at),
             el("mark", {class: span.name, text: text.slice(span.at, span.end)}));
    pos = span.end;
  });
  p.append(text.slice(pos));
}

// Returns one chat bubble with its speaker line and evidence chips.
function bubble(role, label, text, cites, target) {
  const meta = el("div", {class: "meta"}, el("b", {text: label}));
  cites.forEach((cite) => {
    if (cite.label) {
      meta.append(el("span", {class: "chip " + cite.name, text: cite.label}));
    }
  });
  const empty = !text.trim();
  const p = el("p", {class: "bubble" + (empty ? " empty" : "")});
  if (empty) p.textContent = "(empty)";
  else richText(p, text, cites);
  return el("div", {class: "msg " + role, "data-target": target}, meta, p);
}

function cardView() {
  return el("div", {class: "card", "data-target": "scene"},
    el("b", {text: "Character card"}),
    el("span", {
      class: "hint",
      text: ` \u00b7 ${state.scenario_id || "no id"} \u00b7 group ` +
        (state.group || "\u2014"),
    }),
    el("p", {text: state.system.trim() || "(empty)"}));
}

function markSelected() {
  document.querySelectorAll("#chat [data-target]").forEach((node) => {
    node.classList.toggle("sel", node.dataset.target === ui.selected);
  });
}

function markBad() {
  const bad = new Set(ui.showErrors ? problems().map((p) => p.target) : []);
  document.querySelectorAll("#chat [data-target]").forEach((node) => {
    node.classList.toggle("bad", bad.has(node.dataset.target));
  });
}

// Opens target ("scene", "turn:<key>", "ending:<name>") in the sidebar.
function select(target) {
  ui.selected = target;
  renderFields();
  const node = document.querySelector(`#chat [data-target="${target}"]`);
  if (node) node.scrollIntoView({block: "nearest"});
}

// Drags the sidebar width and remembers it for the next visit.
function setupSplit() {
  const split = $("split");
  const narrow = "(max-width: 900px)";
  function applySide(px) {
    const cols = document.querySelector(".cols");
    const room = cols.clientWidth || window.innerWidth;
    const min = 260;
    const max = Math.max(min, room - 280);
    const width = Math.min(max, Math.max(min, px));
    document.documentElement.style.setProperty("--side", width + "px");
  }
  const saved = Number(localStorage.getItem("author-side"));
  if (saved && !matchMedia(narrow).matches) applySide(saved);
  split.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || matchMedia(narrow).matches) return;
    try {
      split.setPointerCapture(event.pointerId);
    } catch (error) {
      // The pointer can already be gone.
    }
    document.body.classList.add("resizing");
    const box = document.querySelector(".cols").getBoundingClientRect();
    function move(ev) { applySide(box.right - ev.clientX); }
    function up() {
      document.body.classList.remove("resizing");
      split.removeEventListener("pointermove", move);
      split.removeEventListener("pointerup", up);
      split.removeEventListener("pointercancel", up);
      const width = getComputedStyle(document.documentElement)
        .getPropertyValue("--side");
      localStorage.setItem("author-side", String(parseInt(width, 10)));
    }
    split.addEventListener("pointermove", move);
    split.addEventListener("pointerup", up);
    split.addEventListener("pointercancel", up);
    event.preventDefault();
  });
}
setupSplit();

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
           hint: GROUP_HINT}),
    field({label: "Character card", value: state.system, rows: 6,
           set: (v) => { state.system = v; },
           hint: CARD_HINT}),
  ];
}

function insertTurn(index, role) {
  const turn = {key: newKey(), role: role, content: "", alt: ""};
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
