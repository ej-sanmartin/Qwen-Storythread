const $ = (id) => document.getElementById(id);
let queue = [];
let scene = null;
let selected = null;
let sortKey = "order";
let sortAsc = true;
let loadToken = 0;

function label(status) {
  return status ? status.charAt(0).toUpperCase() + status.slice(1) : "";
}

function parseIntField(value) {
  if (!/^-?\d+$/.test(String(value).trim())) return null;
  return Number(value);
}

function setNote(text, kind) {
  const note = $("note");
  note.textContent = text || "";
  note.className = kind || "";
}

function setBusy(busy) {
  ["approve", "skip", "pending"].forEach((id) => { $(id).disabled = busy; });
}

async function readJson(response) {
  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error("The reviewer returned an unreadable response.");
  }
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
}

function field(row, key) {
  if (key === "score") return row.score;
  if (key === "range") {
    if (!Number.isInteger(row.suggested_start) ||
        !Number.isInteger(row.suggested_end)) return null;
    return row.suggested_start;
  }
  if (key === "scene_id") return row.scene_id;
  if (key === "name") return row.name || "";
  if (key === "status") return row.status || "";
  return row.order;
}

function compare(a, b) {
  if (sortKey === "order") return a.order - b.order;
  const av = field(a, sortKey);
  const bv = field(b, sortKey);
  const aMissing = av == null || av === "";
  const bMissing = bv == null || bv === "";
  if (aMissing || bMissing) {
    if (aMissing && bMissing) return a.order - b.order;
    return aMissing ? 1 : -1;
  }
  let diff = 0;
  if (typeof av === "number" && typeof bv === "number") {
    diff = av - bv;
    if (sortKey === "range") diff = diff || a.suggested_end - b.suggested_end;
  } else {
    diff = String(av).localeCompare(String(bv));
  }
  if (!sortAsc) diff = -diff;
  return diff || a.order - b.order;
}

function visible() {
  const status = $("filter").value;
  const query = $("query").value.trim().toLowerCase();
  return queue.filter((row) => {
    if (status !== "all" && row.status !== status) return false;
    if (!query) return true;
    const name = (row.name || "").toLowerCase();
    return name.includes(query) || row.scene_id.toLowerCase().includes(query);
  }).slice().sort(compare);
}

function renderQueue() {
  document.querySelectorAll("th button").forEach((button) => {
    let text = button.dataset.label;
    if (button.dataset.sort === sortKey) {
      text += sortAsc ? " \u2191" : " \u2193";
    }
    button.textContent = text;
  });
  const rows = visible();
  $("count").textContent = rows.length + " shown";
  const body = $("queue");
  body.replaceChildren();
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.dataset.id = row.scene_id;
    if (row.scene_id === selected) tr.classList.add("selected");
    const idCell = document.createElement("td");
    const idLink = document.createElement("a");
    idLink.className = "id";
    idLink.href = scenePath(row.scene_id);
    idLink.textContent = row.scene_id.slice(0, 8);
    idLink.title = row.scene_id;
    idCell.append(idLink);
    const name = document.createElement("td");
    name.className = "name";
    name.textContent = row.name || "";
    name.title = row.name || "";
    const status = document.createElement("td");
    status.className = "status " + row.status;
    status.textContent = label(row.status);
    const score = document.createElement("td");
    score.textContent = Number.isInteger(row.score)
      ? String(row.score) : "\u2014";
    if (row.suggestion_count) {
      const extra = row.suggestion_count > 1
        ? " (latest of " + row.suggestion_count + ")" : "";
      score.title = (row.judge_model || "judge") + extra;
    }
    const range = document.createElement("td");
    if (Number.isInteger(row.suggested_start) &&
        Number.isInteger(row.suggested_end)) {
      range.textContent = row.suggested_start + "\u2013" + row.suggested_end;
      range.title = "suggested start " + row.suggested_start +
        ", end " + row.suggested_end + " (exclusive)";
    } else {
      range.textContent = "\u2014";
    }
    tr.append(idCell, name, status, score, range);
    body.append(tr);
  });
}

function describeWindow(turns, start, end) {
  if (start === null || end === null ||
      !(start >= 0 && start < end && end <= turns.length)) {
    return "Range is outside the log. End is exclusive.";
  }
  const ready = end - start >= 2 &&
    turns[end - 2].role === "user" && turns[end - 1].role === "assistant";
  if (ready) {
    return "Window ends with a user turn, then the final assistant reply.";
  }
  return "Window must end with a user turn, then an assistant reply.";
}

function edge(text) {
  const line = document.createElement("div");
  line.className = "edge";
  line.textContent = text;
  return line;
}

function appendRich(parent, text) {
  text.split(/(\*[^*\n]+\*)/).forEach((part) => {
    if (part.length > 2 && part[0] === "*" && part[part.length - 1] === "*") {
      const em = document.createElement("em");
      em.textContent = part.slice(1, -1);
      parent.append(em);
    } else if (part) {
      parent.append(part);
    }
  });
}

function renderLog() {
  const log = $("log");
  log.replaceChildren();
  if (!scene) return;
  const start = parseIntField($("start").value);
  const end = parseIntField($("end").value);
  $("window-note").textContent = describeWindow(scene.turns, start, end);
  const count = scene.turns.length;
  const valid = start !== null && end !== null &&
    start >= 0 && start < end && end <= count;
  let finalRow = null;
  scene.turns.forEach((turn, index) => {
    const inside = valid && index >= start && index < end;
    const isFinal = inside && index === end - 1 && turn.role === "assistant";
    if (valid && index === start) log.append(edge("Window starts at " + start));
    if (valid && index === end) log.append(edge("Window ends before " + end));
    const row = document.createElement("div");
    row.className = "msg " + turn.role + (valid && !inside ? " out" : "") +
      (isFinal ? " final" : "");
    const meta = document.createElement("div");
    meta.className = "meta";
    const who = document.createElement("b");
    const assistant = scene.name || "Assistant";
    who.textContent = turn.role === "user" ? "You" : assistant;
    const tail = isFinal ? " \u00b7 final reply" : "";
    meta.append(who, " \u00b7 " + index + tail);
    const bubble = document.createElement("p");
    bubble.className = "bubble";
    appendRich(bubble, turn.content);
    row.append(meta, bubble);
    log.append(row);
    if (isFinal) finalRow = row;
  });
  if (valid && end === count) log.append(edge("Window ends at " + end));
  if (finalRow) {
    const scroller = finalRow.closest(".read");
    const overflow = getComputedStyle(scroller).overflowY;
    if (overflow === "auto" || overflow === "scroll") {
      finalRow.scrollIntoView({block: "nearest"});
    }
  }
}

function renderSuggestions() {
  const root = $("suggestions");
  root.replaceChildren();
  scene.suggestions.forEach((item) => {
    const block = document.createElement("section");
    block.className = "suggestion";
    const title = document.createElement("h3");
    const bits = [item.judge_model || "judge"];
    if (Number.isInteger(item.total)) bits.push("score " + item.total);
    if (Number.isInteger(item.start) && Number.isInteger(item.end)) {
      bits.push(item.start + "\u2013" + item.end);
    }
    title.append(bits.join(" \u00b7 "));
    if (item.status) {
      title.append(" \u00b7 ");
      const verdict = document.createElement("span");
      verdict.className = "verdict " + item.status;
      verdict.textContent = item.status;
      title.append(verdict);
    }
    block.append(title);
    [
      ["keep_reason", "reason", "reason"],
      ["concern", "concern", "concern"],
      ["skip_reason", "skip reason", "skip"],
    ].forEach(([key, label, kind]) => {
      if (!item[key]) return;
      const paragraph = document.createElement("p");
      paragraph.className = "note-line " + kind;
      const name = document.createElement("b");
      name.textContent = label + ":";
      paragraph.append(name, " " + item[key]);
      block.append(paragraph);
    });
    if (Array.isArray(item.flags) && item.flags.length) {
      const flags = document.createElement("p");
      flags.className = "flags";
      flags.textContent = "flags: " + item.flags.join(", ");
      block.append(flags);
    }
    if (Number.isInteger(item.start) && Number.isInteger(item.end)) {
      const use = document.createElement("button");
      use.type = "button";
      use.textContent = "Use this range";
      use.addEventListener("click", () => {
        $("start").value = String(item.start);
        $("end").value = String(item.end);
        setNote("");
        renderLog();
      });
      block.append(use);
    }
    root.append(block);
  });
}
