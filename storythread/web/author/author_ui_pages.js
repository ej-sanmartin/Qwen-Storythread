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

// Checks the scene, then writes it. Other rows stay untouched.
async function save() {
  ui.showErrors = true;
  ui.serverError = "";
  ui.note = "";
  renderStatus();
  renderProblems();
  if (problems().length) {
    $("side").scrollTop = 0;
    return;
  }
  const oldKey = draftKey();
  const url = state.originalId
    ? "/api/scenes/" + encodeURIComponent(state.originalId) : "/api/scenes";
  $("save").disabled = true;
  try {
    const body = await readJson(await fetch(url, {
      method: "POST", headers: HEADERS, body: JSON.stringify(payload()),
    }));
    applySaved(body.scene);
    state.originalId = body.scene.scenario_id;
    state.dirty = false;
    ui.showErrors = false;
    ui.note = "Saved " + state.originalId + ".";
    dropDraft(oldKey);
    dropDraft(state.originalId);
    history.replaceState({}, "", "/scene/" + encodeURIComponent(state.originalId));
    await loadList();
    renderChat();
    renderFields();
  } catch (error) {
    ui.serverError = error.message;
  } finally {
    $("save").disabled = false;
    renderStatus();
    renderProblems();
  }
}

function discard() {
  if (!state) return;
  if (state.dirty && !confirm("Discard your unsaved changes?")) return;
  dropDraft(draftKey());
  location.reload();
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

// ---- Pages --------------------------------------------------------------

async function loadList() {
  scenes = await readJson(await fetch("/api/scenes"));
}

// Draws the saved-scene table, with unsaved drafts flagged.
function renderList() {
  const rows = [];
  const fresh = loadDraft("new");
  if (fresh) {
    rows.push({id: fresh.scenario_id || "(new)", group: fresh.group,
               turns: fresh.history.length, href: "/new",
               status: "Unsaved new scene", saved: false});
  }
  scenes.forEach((item) => rows.push({
    id: item.scenario_id, group: item.group, turns: item.turns,
    href: "/scene/" + encodeURIComponent(item.scenario_id),
    status: loadDraft(item.scenario_id) ? "Unsaved edits" : "",
    saved: true,
  }));
  $("count").textContent = scenes.length + " saved";
  $("list-empty").hidden = rows.length > 0;
  $("rows").replaceChildren(...rows.map((row) => {
    const tr = el("tr", {class: "row"},
      el("td", {}, el("a", {href: row.href, text: row.id})),
      el("td", {text: row.group}), el("td", {text: String(row.turns)}),
      el("td", {class: "badge", text: row.status}),
      el("td", {}, row.saved ? el("button", {
        type: "button", class: "danger", text: "Delete",
        onclick: (event) => askDelete(event, row.id),
      }) : null));
    tr.addEventListener("click", (event) => {
      if (event.target.closest("a, button")) return;
      location.href = row.href;
    });
    return tr;
  }));
}

// Opens the confirmation modal for one saved scene.
function askDelete(event, id) {
  event.preventDefault();
  event.stopPropagation();
  $("confirm").dataset.id = id;
  $("confirm-text").textContent =
    "Are you sure you want to delete \u201c" + id + "\u201d? " +
    "This removes it from the saved scenes.";
  $("confirm-error").hidden = true;
  $("confirm-error").textContent = "";
  $("confirm").hidden = false;
  $("confirm-cancel").focus();
}

// Closes the confirmation modal without deleting.
function closeConfirm() {
  $("confirm").hidden = true;
  delete $("confirm").dataset.id;
}

// Deletes the scene named in the open modal, then redraws the list.
async function confirmDelete() {
  const id = $("confirm").dataset.id;
  if (!id) return;
  $("confirm-ok").disabled = true;
  $("confirm-error").hidden = true;
  try {
    await readJson(await fetch(
      "/api/scenes/" + encodeURIComponent(id), {method: "DELETE"}));
    dropDraft(id);
    closeConfirm();
    await loadList();
    renderList();
  } catch (error) {
    $("confirm-error").textContent = error.message;
    $("confirm-error").hidden = false;
  } finally {
    $("confirm-ok").disabled = false;
  }
}

// Opens the editor for a saved id, or a new scene when id is null.
async function openEditor(id) {
  const draft = loadDraft(id || "new");
  if (draft) {
    state = draft;
    ui.note = "Restored your unsaved edits.";
  } else if (id) {
    state = fromServer(await readJson(
      await fetch("/api/scenes/" + encodeURIComponent(id))));
  } else {
    state = blankScene();
  }
  $("list-page").hidden = true;
  $("edit").hidden = false;
  const cap = Number(localStorage.getItem("author-cap"));
  $("cap").value = Number.isInteger(cap) && cap > 0 ? cap : 512;
  renderStatus();
  renderChat();
  renderFields();
  renderProblems();
}

async function boot() {
  const path = decodeURIComponent(location.pathname);
  try {
    await loadList();
    if (path === "/new") return await openEditor(null);
    if (path.startsWith("/scene/") && path.length > 7) {
      return await openEditor(path.slice(7));
    }
  } catch (error) {
    $("list-error").textContent = error.message;
  }
  renderList();
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("theme").textContent = theme === "dark" ? "Light" : "Dark";
}

$("new").addEventListener("click", () => { location.href = "/new"; });
$("confirm-cancel").addEventListener("click", closeConfirm);
$("confirm-ok").addEventListener("click", confirmDelete);
$("confirm").addEventListener("click", (event) => {
  if (event.target === $("confirm")) closeConfirm();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("confirm").hidden) closeConfirm();
});
$("save").addEventListener("click", save);
$("discard").addEventListener("click", discard);
$("check").addEventListener("click", checkTokens);
$("cap").addEventListener("input", () => {
  const cap = Number($("cap").value);
  if (Number.isInteger(cap) && cap > 0) {
    localStorage.setItem("author-cap", String(cap));
  }
});
$("chat").addEventListener("click", (event) => {
  const host = event.target.closest("[data-target]");
  // A bubble that is already open keeps the user's text highlight.
  if (host && state && host.dataset.target !== ui.selected) {
    select(host.dataset.target);
  }
});
// Keeps the highlighted words while a sidebar button is pressed.
$("side").addEventListener("mousedown", (event) => {
  if (event.target.closest("[data-keepsel]")) event.preventDefault();
});
const storedTheme = localStorage.getItem("author-theme");
const systemTheme = matchMedia("(prefers-color-scheme: dark)").matches;
applyTheme(storedTheme || (systemTheme ? "dark" : "light"));
$("theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark"
    ? "light" : "dark";
  localStorage.setItem("author-theme", next);
  applyTheme(next);
});
boot();
