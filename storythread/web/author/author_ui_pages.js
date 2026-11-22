// ---- Save, discard, and token check -------------------------------------

// Copies the server's stripped text back, keeping turn keys.
function applySaved(scene) {
  state.scenario_id = scene.scenario_id;
  state.group = scene.group;
  state.system = scene.system;
  scene.history.forEach((turn, i) => {
    if (!state.history[i]) return;
    state.history[i].role = turn.role;
    state.history[i].content = turn.content;
  });
  NAMES.forEach((name) => {
    const raw = scene.branches[name];
    Object.assign(state.branches[name], {
      user: raw.user, reply: raw.reply, evidence: raw.evidence || "",
    });
  });
}

// Shows preflight counts and flags for both endings.
function renderTokens(report) {
  $("tokens").replaceChildren(...NAMES.map((name) => {
    const info = report[name];
    const flags = info.flags.length
      ? "flags: " + info.flags.join(", ") : "no flags";
    return el("p", {
      class: info.flags.length ? "bad-text" : "",
      text: `${TITLE[name]}: ${info.full} tokens (${info.input} in, ` +
        `${info.target} reply) \u00b7 evidence ${plural(info.distance_turns)}` +
        `, ${info.evidence_distance_tokens} tokens back \u00b7 ${flags}`,
    });
  }));
}

// Measures both endings against the cap in the toolbar.
async function checkTokens() {
  const cap = Number($("cap").value);
  if (!Number.isInteger(cap) || cap < 1) {
    ui.serverError = "Token cap must be a positive integer.";
    renderProblems();
    return;
  }
  ui.serverError = "";
  $("check").disabled = true;
  try {
    const report = await readJson(await fetch("/api/tokens", {
      method: "POST", headers: HEADERS,
      body: JSON.stringify({max_tokens: cap, ...payload()}),
    }));
    renderTokens(report);
  } catch (error) {
    $("tokens").replaceChildren();
    ui.serverError = error.message;
  } finally {
    $("check").disabled = false;
    renderProblems();
  }
}

$("check").addEventListener("click", checkTokens);
$("cap").addEventListener("input", () => {
  const cap = Number($("cap").value);
  if (Number.isInteger(cap) && cap > 0) {
    localStorage.setItem("author-cap", String(cap));
  }
});
