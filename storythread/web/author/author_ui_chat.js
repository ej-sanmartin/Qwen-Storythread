// ---- Chat ---------------------------------------------------------------

// Returns an evidence chip label such as "Recent evidence \u00b7 3 turns back".
function chipLabel(name, back) {
  return `${TITLE[name]} \u00b7 ${plural(back)} back`;
}

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

function endingView(name) {
  const b = state.branches[name];
  const index = evidenceIndex(b);
  const own = b.source === "self" && !b.missing
    ? [{name: name, quote: b.evidence, label: chipLabel(name, 1)}] : [];
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
      (cite) => ({...cite, label: chipLabel(cite.name, total + 1 - i)}));
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
