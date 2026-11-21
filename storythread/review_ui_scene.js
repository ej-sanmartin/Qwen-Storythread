function scenePath(id) {
  return "/scene/" + encodeURIComponent(id);
}

function sceneFromLocation() {
  const path = decodeURIComponent(location.pathname);
  const prefix = "/scene/";
  if (!path.startsWith(prefix) || path === prefix) return null;
  return path.slice(prefix.length);
}

// Shows the scene full-page on a narrow screen.
function markScene(on) {
  document.body.classList.toggle("scene", on);
}

function showQueue() {
  loadToken += 1;
  selected = null;
  scene = null;
  markScene(false);
  $("editor").hidden = true;
  $("empty").hidden = false;
  $("empty").textContent = "Select a scene.";
  document.title = "Storythread Review";
  renderQueue();
}

async function go(id) {
  const url = scenePath(id);
  if (decodeURIComponent(location.pathname) !== url) {
    history.pushState({scene_id: id}, "", url);
  }
  await openScene(id);
}

async function openScene(id) {
  const token = ++loadToken;
  selected = id;
  markScene(true);
  renderQueue();
  setNote("");
  try {
    const response = await fetch("/api/scene/" + encodeURIComponent(id));
    const payload = await readJson(response);
    if (token !== loadToken) return;
    scene = payload;
  } catch (error) {
    if (token !== loadToken) return;
    $("editor").hidden = true;
    $("empty").hidden = false;
    $("empty").textContent = error.message;
    return;
  }
  $("empty").hidden = true;
  $("editor").hidden = false;
  $("name").textContent = scene.name || "(unnamed)";
  $("chat-who").textContent = scene.name || "(unnamed)";
  const status = $("status");
  status.textContent = label(scene.status);
  status.className = "status " + scene.status;
  $("group").textContent = scene.group || "(none)";
  $("scene-id").textContent = scene.scene_id;
  document.title = (scene.name || "Scene") + " · Storythread Review";
  $("system").value = scene.system || "";
  $("start").value = Number.isInteger(scene.start) ? String(scene.start) : "";
  $("end").value = Number.isInteger(scene.end) ? String(scene.end) : "";
  $("length").textContent = scene.turns.length + " turns";
  renderSuggestions();
  renderLog();
}

async function loadQueue() {
  queue = await readJson(await fetch("/api/queue"));
  renderQueue();
}

async function decide(status, advance) {
  if (!scene) return;
  const start = parseIntField($("start").value);
  const end = parseIntField($("end").value);
  if (start === null || end === null) {
    setNote("Start and end must be integers.", "bad");
    return;
  }
  const previous = visible().map((row) => row.scene_id);
  const index = previous.indexOf(selected);
  setBusy(true);
  setNote("");
  try {
    await readJson(await fetch("/api/scene/" + encodeURIComponent(selected), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        system: $("system").value,
        start: start,
        end: end,
        review_status: status,
      }),
    }));
    await loadQueue();
    let pick = null;
    if (advance && index >= 0) {
      const still = new Set(visible().map((row) => row.scene_id));
      for (let i = index + 1; i < previous.length; i += 1) {
        if (still.has(previous[i])) { pick = previous[i]; break; }
      }
    }
    if (pick) {
      await go(pick);
    } else {
      await openScene(selected);
      if (advance) setNote("Saved. No further rows in this queue.", "ok");
    }
  } catch (error) {
    setNote(error.message, "bad");
  } finally {
    setBusy(false);
  }
}

function setSort(key) {
  if (sortKey === key) sortAsc = !sortAsc;
  else {
    sortKey = key;
    sortAsc = key !== "score";
  }
  renderQueue();
}

document.querySelectorAll("th button").forEach((button) => {
  button.dataset.label = button.textContent.trim();
  button.addEventListener("click", () => setSort(button.dataset.sort));
});
$("queue").addEventListener("click", (event) => {
  const row = event.target.closest("tr");
  if (!row || !row.dataset.id) return;
  if (event.button !== 0 || event.metaKey || event.ctrlKey ||
      event.shiftKey || event.altKey) return;
  event.preventDefault();
  go(row.dataset.id);
});
window.addEventListener("popstate", () => {
  const id = sceneFromLocation();
  if (id) openScene(id);
  else showQueue();
});
$("back").addEventListener("click", (event) => {
  if (event.button !== 0 || event.metaKey || event.ctrlKey ||
      event.shiftKey || event.altKey) return;
  event.preventDefault();
  if (location.pathname !== "/") history.pushState({}, "", "/");
  showQueue();
});
// Drags the queue width and remembers it.
function setupSplit() {
  const split = $("split");
  const narrow = "(max-width: 800px)";
  function applyAside(px) {
    const panes = document.querySelector(".panes");
    const min = 220;
    const max = Math.max(min, panes.clientWidth - 280);
    const width = Math.min(max, Math.max(min, px));
    const root = document.documentElement;
    root.style.setProperty("--aside", width + "px");
    root.style.setProperty("--aside-max", "none");
  }
  const saved = Number(localStorage.getItem("review-aside"));
  if (saved && !matchMedia(narrow).matches) applyAside(saved);
  split.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || matchMedia(narrow).matches) return;
    try {
      split.setPointerCapture(event.pointerId);
    } catch (error) {
      // The pointer can already be gone.
    }
    document.body.classList.add("resizing");
    const left = document.querySelector(".panes").getBoundingClientRect().left;
    function move(ev) { applyAside(ev.clientX - left); }
    function up() {
      document.body.classList.remove("resizing");
      split.removeEventListener("pointermove", move);
      split.removeEventListener("pointerup", up);
      split.removeEventListener("pointercancel", up);
      const width = getComputedStyle(document.documentElement)
        .getPropertyValue("--aside");
      localStorage.setItem("review-aside", String(parseInt(width, 10)));
    }
    split.addEventListener("pointermove", move);
    split.addEventListener("pointerup", up);
    split.addEventListener("pointercancel", up);
    event.preventDefault();
  });
}
setupSplit();
// Drags the chat height and remembers it.
function setupChatSplit() {
  const split = $("split-y");
  const above = document.querySelector(".above");
  function applyAbove(px) {
    const editor = $("editor");
    const room = editor.clientHeight || window.innerHeight - 80;
    const min = 120;
    const max = Math.max(min, room - 140);
    const height = Math.min(max, Math.max(min, px));
    const root = document.documentElement;
    root.style.setProperty("--above", height + "px");
    root.style.setProperty("--above-max", "none");
  }
  const saved = Number(localStorage.getItem("review-above"));
  if (saved) applyAbove(saved);
  split.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    try {
      split.setPointerCapture(event.pointerId);
    } catch (error) {
      // The pointer can already be gone.
    }
    document.body.classList.add("resizing-y");
    const top = above.getBoundingClientRect().top;
    function move(ev) { applyAbove(ev.clientY - top); }
    function up() {
      document.body.classList.remove("resizing-y");
      split.removeEventListener("pointermove", move);
      split.removeEventListener("pointerup", up);
      split.removeEventListener("pointercancel", up);
      const height = getComputedStyle(document.documentElement)
        .getPropertyValue("--above");
      localStorage.setItem("review-above", String(parseInt(height, 10)));
    }
    split.addEventListener("pointermove", move);
    split.addEventListener("pointerup", up);
    split.addEventListener("pointercancel", up);
    event.preventDefault();
  });
}
setupChatSplit();
$("filter").addEventListener("change", renderQueue);
$("query").addEventListener("input", renderQueue);
function onWindowEdit() {
  setNote("");
  renderLog();
}
$("start").addEventListener("input", onWindowEdit);
$("end").addEventListener("input", onWindowEdit);
$("approve").addEventListener("click", () => decide("approved", true));
$("skip").addEventListener("click", () => decide("skipped", true));
$("pending").addEventListener("click", () => decide("pending", false));

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("theme").textContent = theme === "dark" ? "Light" : "Dark";
}
const stored = localStorage.getItem("review-theme");
const system = matchMedia("(prefers-color-scheme: dark)").matches;
applyTheme(stored || (system ? "dark" : "light"));
$("theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark"
    ? "light" : "dark";
  localStorage.setItem("review-theme", next);
  applyTheme(next);
});

$("count").textContent = "Loading…";
loadQueue().then(() => {
  const id = sceneFromLocation();
  if (id) return openScene(id);
}).catch((error) => {
  $("empty").textContent = error.message;
});
