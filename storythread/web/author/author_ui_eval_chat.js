// ---- Chat (eval) --------------------------------------------------------

// Returns one history turn as bubbles: two for the evidence turn.
function turnViews(turn, i) {
  const label = `${i} \u00b7 ${SPEAKER[turn.role] || turn.role}`;
  const target = "turn:" + turn.key;
  if (turn.key !== state.evidence) {
    return [bubble(turn.role, label, turn.content, [], target)];
  }
  return VERSIONS.map((name) => bubble(
    turn.role, label, name === "a" ? turn.content : turn.alt,
    [{name: name, quote: state.quotes[name],
      label: `${TITLE[name]} \u00b7 ${plural(turnsBack())} back`}],
    target));
}

// Returns what a correct reply does for one version.
function expectedView(name) {
  const quote = state.quotes[name].trim();
  return el("section", {class: "ending", "data-target": "expected:" + name},
    el("h3", {text: TITLE[name]}),
    el("p", {class: "expected", text: state.expected[name].trim() || "(empty)"}),
    el("p", {class: "ev", text: quote ? `Evidence: \u201c${quote}\u201d`
                                      : "Evidence quote not set yet"}));
}

// Rebuilds the whole chat from state.
function renderChat() {
  const log = el("div", {class: "log"},
    ...state.history.flatMap((turn, i) => turnViews(turn, i)));
  const final = el("div", {class: "log"},
    bubble("user", "Player \u00b7 final turn", state.final, [], "final"));
  $("chat").replaceChildren(
    cardView(), log,
    el("button", {type: "button", class: "add-turn", text: "Add turn",
                  onclick: addTurn}),
    el("div", {class: "edge", text: "Final turn, the same in both versions"}),
    final,
    el("div", {class: "edge", text: "What a correct reply does"}),
    el("div", {class: "endings"}, ...VERSIONS.map(expectedView)));
  markSelected();
  markBad();
}
