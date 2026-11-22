const $ = (id) => document.getElementById(id);
const HEADERS = {"Content-Type": "application/json"};
const SPEAKER = {assistant: "Character", user: "Player"};
let sheets = [];
let sheet = null;
let cases = [];
let index = 0;

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

// Returns parsed JSON or throws the server's error text.
async function readJson(response) {
  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error("The server returned an unreadable response.");
  }
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
}

// Returns true when a case already has every rating it needs.
function isRated(row) {
  if (sheet.kind === "continuity") {
    return row.replies.every((r) => typeof r.satisfies === "boolean");
  }
  return ["A", "B", "tie"].includes(row.winner);
}

function tabLabel(item) {
  const name = item.kind === "continuity" ? "Continuity"
    : item.name.replace(/^blind-/, "").replace("-", " vs ");
  return `${name} \u00b7 ${item.rated}/${item.total}`;
}

// Redraws the sheet tabs and the overall progress line.
function renderTabs() {
  $("tabs").replaceChildren(...sheets.map((item) => el("button", {
    type: "button", text: tabLabel(item),
    class: (sheet && sheet.name === item.name ? "on " : "") +
      (item.rated === item.total ? "done" : ""),
    onclick: () => openSheet(item.name),
  })));
  const rated = sheets.reduce((n, s) => n + s.rated, 0);
  const total = sheets.reduce((n, s) => n + s.total, 0);
  $("progress").textContent = `${rated} of ${total} ratings done`;
}

// Returns a chat bubble; quote (optional) is underlined inside text.
function bubble(role, label, text, quote, evidence) {
  const p = el("p", {class: "bubble"});
  const at = quote ? text.indexOf(quote) : -1;
  if (at < 0) p.textContent = text;
  else p.append(text.slice(0, at), el("mark", {text: quote}),
                text.slice(at + quote.length));
  return el("div", {class: "msg " + role + (evidence ? " evidence" : "")},
            el("div", {class: "meta", text: label}), p);
}

// Returns the card, the story so far, and the evidence note if there is one.
function story(row) {
  const [card, ...turns] = row.messages;
  const log = el("div", {class: "log"}, ...turns.map((turn, i) => bubble(
    turn.role, `${i} \u00b7 ${SPEAKER[turn.role] || turn.role}`, turn.content,
    i === row.evidence_turn ? row.evidence : null, i === row.evidence_turn)));
  const kids = [
    el("details", {class: "card"}, el("summary", {text: "Character card"}),
       el("p", {text: card.content})),
    log,
  ];
  if (row.expected) {
    kids.push(el("div", {class: "box"}, el("b", {text: "Expected"}),
                 row.expected));
    kids.push(el("div", {class: "box"}, el("b", {text: "Evidence"}),
                 `\u201c${row.evidence}\u201d \u00b7 turn ${row.evidence_turn}`));
  }
  return kids;
}

// Saves the open case, then refreshes the progress counts.
async function save(row, payload) {
  try {
    await readJson(await fetch("/api/sheets/" + encodeURIComponent(sheet.name), {
      method: "POST", headers: HEADERS,
      body: JSON.stringify({id: row.id, ...payload}),
    }));
    $("error").hidden = true;
    sheets = await readJson(await fetch("/api/sheets"));
    sheet = sheets.find((item) => item.name === sheet.name);
    renderTabs();
    render();
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
  }
}

// Returns Yes/No buttons for one continuity reply; a second click clears.
function replyView(row, reply) {
  const choose = (value) => {
    reply.satisfies = reply.satisfies === value ? null : value;
    save(row, {replies: row.replies.map(
      (r) => ({label: r.label, satisfies: r.satisfies}))});
  };
  return el("div", {class: "reply"}, el("h3", {text: reply.label}),
    el("p", {text: reply.text}),
    el("div", {class: "btns"},
      el("button", {type: "button", text: "Satisfies expected",
                    class: "yes" + (reply.satisfies === true ? " on" : ""),
                    onclick: () => choose(true)}),
      el("button", {type: "button", text: "Does not",
                    class: "no" + (reply.satisfies === false ? " on" : ""),
                    onclick: () => choose(false)})));
}

// Returns the A / B / tie controls for one engagement case.
function pairView(row) {
  const choose = (value) => save(row, {
    winner: row.winner === value ? "" : value});
  const reply = (side) => el("div", {class: "reply"}, el("h3", {text: side}),
    el("p", {text: row[side]}));
  const pick = (value, text) => el("button", {
    type: "button", text: text, class: row.winner === value ? "on" : "",
    onclick: () => choose(value)});
  return [
    el("p", {class: "hint", text: row.question}),
    el("div", {class: "pair"}, reply("A"), reply("B")),
    el("div", {class: "btns"}, pick("A", "A"), pick("tie", "Tie"),
       pick("B", "B")),
  ];
}

// Draws the open case.
function render() {
  const row = cases[index];
  $("where").textContent = `Case ${index + 1} of ${cases.length} \u00b7 ` +
    `${row.id}` + (isRated(row) ? " \u00b7 rated" : "");
  $("prev").disabled = index === 0;
  $("next").disabled = index === cases.length - 1;
  $("unrated").disabled = cases.every(isRated);
  $("unrated").classList.toggle("on", isRated(row) && !cases.every(isRated));
  const body = sheet.kind === "continuity"
    ? el("div", {class: "replies"}, ...row.replies.map((r) => replyView(row, r)))
    : el("div", {}, ...pairView(row));
  $("view").replaceChildren(...story(row), body);
}

// Opens a sheet at its first unrated case.
async function openSheet(name) {
  sheet = sheets.find((item) => item.name === name);
  cases = await readJson(await fetch("/api/sheets/" + encodeURIComponent(name)));
  const first = cases.findIndex((row) => !isRated(row));
  index = first < 0 ? 0 : first;
  renderTabs();
  render();
}

function step(delta) {
  index = Math.min(cases.length - 1, Math.max(0, index + delta));
  render();
}

function nextUnrated() {
  for (let n = 1; n <= cases.length; n += 1) {
    const at = (index + n) % cases.length;
    if (!isRated(cases[at])) {
      index = at;
      render();
      return;
    }
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("theme").textContent = theme === "dark" ? "Light" : "Dark";
}

async function boot() {
  try {
    sheets = await readJson(await fetch("/api/sheets"));
    if (!sheets.length) throw new Error("No rating sheets found.");
    await openSheet((sheets.find((s) => s.rated < s.total) || sheets[0]).name);
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
  }
}

$("prev").addEventListener("click", () => step(-1));
$("next").addEventListener("click", () => step(1));
$("unrated").addEventListener("click", nextUnrated);
const storedTheme = localStorage.getItem("rate-theme");
const systemTheme = matchMedia("(prefers-color-scheme: dark)").matches;
applyTheme(storedTheme || (systemTheme ? "dark" : "light"));
$("theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark"
    ? "light" : "dark";
  localStorage.setItem("rate-theme", next);
  applyTheme(next);
});
boot();
