// ---- Chat ---------------------------------------------------------------

// Returns {turn key or "self": [{name, quote}]} for both endings.
function citations() {
  const map = {};
  NAMES.forEach((name) => {
    const b = state.branches[name];
    if (!b.source || b.missing) return;
    (map[b.source] = map[b.source] || []).push({name: name, quote: b.evidence});
  });
  return map;
}

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
  cites.forEach((cite) => meta.append(el("span", {
    class: "chip " + cite.name,
    text: `${TITLE[cite.name]} \u00b7 ${plural(cite.back)} back`,
  })));
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

function endingView(name) {
  const b = state.branches[name];
  const index = evidenceIndex(b);
  const own = b.source === "self" && !b.missing
    ? [{name: name, quote: b.evidence, back: 1}] : [];
  let line = "Evidence: not chosen yet";
  if (b.missing) line = "Cited turn was removed.";
  else if (index !== null) {
    const where = index === state.history.length
      ? "this ending's user turn" : "turn " + index;
    line = `Evidence: ${where} \u00b7 ${plural(turnsBack(b))} back`;
  }
  return el("section", {
    class: "ending" + (b.missing ? " missing" : ""),
    "data-target": "ending:" + name,
  },
    el("h3", {text: TITLE[name]}), el("p", {class: "hint", text: HINT[name]}),
    bubble("user", "Player \u00b7 final turn", b.user, own, null),
    bubble("assistant", "Character \u00b7 ideal reply", b.reply, [], null),
    el("p", {class: "ev" + (b.missing ? " bad-text" : ""), text: line}));
}

// Rebuilds the whole chat from state.
function renderChat() {
  const cites = citations();
  const total = state.history.length;
  const log = el("div", {class: "log"}, ...state.history.map((turn, i) => {
    const mine = (cites[turn.key] || []).map(
      (cite) => ({...cite, back: total + 1 - i}));
    return bubble(turn.role, `${i} \u00b7 ${SPEAKER[turn.role] || turn.role}`,
                  turn.content, mine, "turn:" + turn.key);
  }));
  $("chat").replaceChildren(
    cardView(), log,
    el("button", {type: "button", class: "add-turn", text: "Add turn",
                  onclick: addTurn}),
    el("div", {class: "edge", text: "Two endings to the same story"}),
    el("div", {class: "endings"}, ...NAMES.map(endingView)));
  markSelected();
  markBad();
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
