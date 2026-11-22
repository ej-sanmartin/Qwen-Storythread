// Details card, trend chart, and glossary for the adapter report. Shares
// globals with inspect_adapters.js (REPORT, state, COLORS, valueAt, fmt,
// signed, el, svg, $), the glossary file, and inspect_adapters_3d.js.

// Returns the line chart of one module's RMS across the saved steps.
function chart(path) {
  const W = 520, H = 190, L = 56, R = 14, T = 12, B = 30;
  const series = REPORT.arms.map((arm) => ({arm, points: REPORT.steps
    .filter((s) => valueAt(arm, s, path) !== undefined)
    .map((s) => [s, valueAt(arm, s, path)])}));
  const top = Math.max(1e-12, ...series.flatMap((s) => s.points.map((p) => p[1])));
  const lo = REPORT.steps[0], hi = REPORT.steps[REPORT.steps.length - 1];
  const x = (s) => L + (hi === lo ? 0.5 : (s - lo) / (hi - lo)) * (W - L - R);
  const y = (v) => T + (1 - v / top) * (H - T - B);
  const line = (x1, y1, x2, y2, extra = {}) =>
    svg("line", {x1, y1, x2, y2, stroke: "#a1a1aa", ...extra});
  const root = svg("svg", {viewBox: `0 0 ${W} ${H}`, class: "chart",
    role: "img", "aria-label": "Effective LoRA update RMS across steps"},
    line(L, y(0), W - R, y(0)), line(L, T, L, y(0)),
    svg("text", {x: L - 6, y: y(0) + 3, "text-anchor": "end"}, "0"),
    svg("text", {x: L - 6, y: T + 3, "text-anchor": "end"}, fmt(top)),
    line(x(state.step), T, x(state.step), y(0),
         {"stroke-dasharray": "3 3"}));
  for (const s of REPORT.steps) {
    root.append(svg("text", {x: x(s), y: H - 10, "text-anchor": "middle"}, `${s}`));
  }
  for (const {arm, points} of series) {
    const color = COLORS[arm] || "#18181b";
    root.append(svg("polyline", {fill: "none", stroke: color, "stroke-width": 2,
      points: points.map(([s, v]) => `${x(s)},${y(v)}`).join(" ")}));
    for (const [s, v] of points) {
      root.append(svg("circle", {cx: x(s), cy: y(v), r: 3.5, fill: color}));
    }
  }
  return root;
}

// Returns the per-step RMS table for one module.
function stepTable(path) {
  const head = el("tr", {}, el("th", {}, "Arm"),
    ...REPORT.steps.map((s) => el("th", {}, `step ${s}`)));
  const rows = REPORT.arms.map((arm) => el("tr", {},
    el("td", {}, arm), ...REPORT.steps.map((s) => {
      const v = valueAt(arm, s, path);
      return el("td", {}, v === undefined ? "\u2014" : v.toPrecision(3));
    })));
  return el("table", {}, head, ...rows);
}

// Returns a swatch key for the arms, matching the chart lines.
function armKey() {
  return el("p", {class: "key"}, ...REPORT.arms.flatMap((arm) => [
    el("span", {class: "swatch",
      style: `background:${COLORS[arm] || "#374151"}`}), `${arm}  `]));
}

// Returns "2560 x 32, rank 8", or per-arm text when the arms differ.
function dimsText(info) {
  const parts = REPORT.arms.filter((arm) => info.dims[arm]).map((arm) => {
    const d = info.dims[arm];
    return [arm, `${d.in} \u00d7 ${d.out}, rank ${d.rank}`];
  });
  return parts.every(([, text]) => text === parts[0][1]) ? parts[0][1]
    : parts.map(([arm, text]) => `${arm}: ${text}`).join("; ");
}

// Fills the compact details card for the clicked module.
function renderDetail() {
  const box = $("detail");
  const path = state.module;
  if (!path) {
    box.replaceChildren(el("p", {class: "hint"},
      "Click a bar or cell to see its module path, what the projection " +
      "does, the checkpoint step, and the effective LoRA update RMS."));
    return;
  }
  const info = REPORT.modules[path];
  const about = describe(info.column);  // null unless the architecture is known
  const block = info.block === null ? "" : `block ${info.block}`;
  const tag = [about && about.group, block].filter(Boolean).join(" \u00b7 ");
  const metrics = REPORT.arms.map((arm) => {
    const v = valueAt(arm, state.step, path);
    return el("div", {class: "metric"},
      el("span", {class: "swatch",
        style: `background:${COLORS[arm] || "#374151"}`}),
      el("span", {}, arm),
      el("strong", {}, v === undefined ? "\u2014" : v.toPrecision(4)));
  });
  const pct = diffPercent(path, state.step);
  const [a, b] = REPORT.arms;
  const gap = pct === undefined ? [] : [el("p", {class: "gap"},
    `${a} is ${signed(pct)} vs ${b}.`)];
  const summary = el("div", {},
    el("h2", {}, COLUMN_LABEL.get(info.column),
      ...(tag ? [el("span", {class: "tag"}, tag)] : [])),
    el("code", {class: "path"}, path),
    el("p", {class: about ? "desc" : "desc muted"}, about ? about.text
      : "No description: this architecture is not in the glossary."),
    el("p", {class: "step"},
      `Checkpoint step ${state.step} \u00b7 Effective LoRA update RMS`),
    el("div", {class: "metrics"}, ...metrics), ...gap,
    el("p", {class: "dims"}, `Matrix ${dimsText(info)}`));
  const trend = el("div", {}, el("h3", {}, "Across saved steps"),
    chart(path), armKey(), stepTable(path));
  box.replaceChildren(el("div", {class: "card"}, summary, trend));
}

// Shows the glossary table when the architecture is known, else a note.
function renderGlossary() {
  $("glossary-known").hidden = !ARCHITECTURE_KNOWN;
  $("glossary-unknown").hidden = ARCHITECTURE_KNOWN;
  if (!ARCHITECTURE_KNOWN) return;
  const head = el("tr", {}, ...["Group", "Label", "Description"].map(
    (name) => el("th", {scope: "col"}, name)));
  const rows = GLOSSARY.map(([prefix, label, text]) =>
    el("tr", {"data-column": `${prefix}.${label}`, tabindex: "-1"},
      el("td", {}, GROUPS[prefix]), el("td", {}, el("code", {}, label)),
      el("td", {}, text)));
  $("glossary-table").replaceChildren(head, ...rows);
  renderBlocks();
}

// Fills the transformer-layer table with one row per adapted block. Layer
// type and projections come from the checkpoint's own tensor paths.
function renderBlocks() {
  const head = el("tr", {}, ...["Block", "Layer type", "What it does",
    "Adapted projections"].map((name) => el("th", {scope: "col"}, name)));
  const rows = REPORT.blocks.filter((b) => b !== null).map((block) => {
    const columns = REPORT.columns.filter((c) => PATHS.has(`${block}|${c.id}`));
    const kind = layerKind(columns.map((c) => c.id));
    const names = columns.flatMap((c, i) =>
      [...(i ? [", "] : []), el("code", {}, c.label)]);
    return el("tr", {"data-column": `block:${block}`, tabindex: "-1"},
      el("td", {}, `block ${block}`), el("td", {}, kind ? kind.group : "\u2014"),
      el("td", {}, kind ? kind.text : "\u2014"), el("td", {}, ...names));
  });
  $("blocks-table").replaceChildren(head, ...rows);
}

// Opens the glossary, scrolls to the rows (projections or blocks) for the
// given ids, and briefly highlights them. Unknown ids are ignored.
function goToGlossary(ids) {
  const rows = [...$("glossary").querySelectorAll("tr[data-column]")]
    .filter((row) => ids.includes(row.dataset.column));
  if (!rows.length) return;
  $("glossary").open = true;
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  rows[0].scrollIntoView({block: "center", behavior: calm ? "auto" : "smooth"});
  rows[0].focus({preventScroll: true});
  for (const row of rows) {
    row.classList.remove("flash");
    void row.offsetWidth;  // Restarts the animation on repeat clicks.
    row.classList.add("flash");
    setTimeout(() => row.classList.remove("flash"), 1800);
  }
}
