"use strict";
// Adapter heatmap page. All data comes from the embedded #report JSON.
const REPORT = JSON.parse(document.getElementById("report").textContent);
const COLORS = {recent: "#3d6a9e", earlier: "#b7791f"};
const SVG_NS = "http://www.w3.org/2000/svg";
const PATHS = new Map();  // "block|column" -> module path
for (const [path, info] of Object.entries(REPORT.modules)) {
  PATHS.set(`${info.block}|${info.column}`, path);
}
const state = {step: REPORT.steps[REPORT.steps.length - 1], module: null,
               view: "size"};  // view: size | diff | flat
const $ = (id) => document.getElementById(id);

// Builds an element: make(namespace, tag, {attr: value}, ...children).
function make(ns, tag, props, kids) {
  const node = ns ? document.createElementNS(ns, tag)
                  : document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  node.append(...kids);
  return node;
}
const el = (tag, props = {}, ...kids) => make(null, tag, props, kids);
const svg = (tag, props = {}, ...kids) => make(SVG_NS, tag, props, kids);

// Returns the RMS for an arm, step, and module path, or undefined.
function valueAt(arm, step, path) {
  const snapshot = REPORT.rms[arm][String(step)];
  return snapshot && path in snapshot ? snapshot[path] : undefined;
}

// Formats an RMS with two significant digits.
function fmt(value) {
  return value === 0 ? "0" : value.toPrecision(2);
}

// Formats a percent with an explicit sign.
function signed(pct) {
  return `${pct >= 0 ? "+" : "\u2212"}${Math.abs(pct).toFixed(1)}%`;
}

// Returns the {hue, sat, light} color for t in 0..1 on the shared scale.
function ramp(t) {
  return {hue: 212, sat: 40 + 10 * t, light: 90 - 50 * t};
}

// Returns the cell colors for a value on the one shared color scale.
function shade(value) {
  const t = Math.min(1, value / (REPORT.max || 1));
  const c = ramp(t);
  return `background:hsl(${c.hue} ${c.sat}% ${c.light}%);` +
         `color:${t > 0.5 ? "#fff" : "#18181b"}`;
}

// Returns the heatmap cell for one arm, block, and projection column.
function cell(arm, block, column) {
  const path = PATHS.get(`${block}|${column.id}`);
  const value = path ? valueAt(arm, state.step, path) : undefined;
  if (value === undefined) return el("span", {class: "cell empty"});
  const selected = path === state.module ? " selected" : "";
  const label = block === null ? "other" : `block ${block}`;
  return el("button", {
    class: `cell${selected}`, style: shade(value), title: path,
    "aria-label": `${arm} ${label} ${column.label}: ${value}`,
    onclick: () => { state.module = path; render(); },
  }, fmt(value));
}

// Returns one arm's heatmap panel for the selected step.
function panel(arm) {
  const present = String(state.step) in REPORT.rms[arm];
  const grid = el("div", {class: "grid", style: `--cols:${REPORT.columns.length}`},
    el("span"),
    ...REPORT.columns.map((c) => el("span", {class: "col", title: c.id}, c.label)));
  for (const block of REPORT.blocks) {
    grid.append(el("span", {class: "row"}, block === null ? "other" : `${block}`));
    for (const column of REPORT.columns) grid.append(cell(arm, block, column));
  }
  const title = el("h2", {}, arm, " ",
    el("span", {}, `scale ${REPORT.scales[arm]}`));
  const absent = present ? [] : [el("p", {class: "absent"},
    `No snapshot for ${arm} at step ${state.step}.`)];
  return el("section", {class: "panel"}, title, ...absent, grid);
}

// Fills the color legend for the current view.
function renderLegend() {
  const [a, b] = REPORT.arms;
  if (state.view === "diff") {
    $("legend").replaceChildren(`${b} larger \u2212${DIFF_MAX}%`,
      el("span", {class: "ramp diverging"}),
      `${a} larger +${DIFF_MAX}% (of ${b}; same scale for all steps)`);
    return;
  }
  $("legend").replaceChildren("0", el("span", {class: "ramp"}),
    `${fmt(REPORT.max)} (same scale for both arms and all steps)`);
}

// Redraws the panels, legend, and detail section for the current view.
function render() {
  scenes.length = 0;
  const panels = state.view === "flat" ? REPORT.arms.map(panel)
    : state.view === "size" ? REPORT.arms.map(sizePanel) : [diffPanel()];
  $("panels").className = state.view === "diff" ? "one" : "two";
  $("panels").replaceChildren(...panels);
  renderLegend();
  drawScenes();
  renderDetail();
}

// Fills the step selector, notes, and legend, then draws.
function boot() {
  const select = $("step");
  for (const step of REPORT.steps) {
    const gaps = REPORT.arms.filter((a) => REPORT.missing[a].includes(step));
    const note = gaps.length ? ` (missing: ${gaps.join(", ")})` : "";
    select.append(el("option", {value: step}, `${step}${note}`));
  }
  select.value = state.step;
  select.addEventListener("change", () => {
    state.step = Number(select.value);
    render();
  });
  const notes = [...REPORT.notes];
  for (const arm of REPORT.arms) {
    if (REPORT.missing[arm].length) {
      notes.push(`${arm} has no snapshot at step(s) ${REPORT.missing[arm].join(", ")}.`);
    }
  }
  $("notes").replaceChildren(...notes.map((n) => el("li", {}, n)));
  $("view").addEventListener("change", (e) => {
    state.view = e.target.value;
    render();
  });
  $("reset").addEventListener("click", () => {
    Object.assign(camera, HOME);
    drawScenes();
  });
  window.addEventListener("resize", drawScenes);
  renderGlossary();
  render();
}
// Runs after every script in the page has loaded.
window.addEventListener("DOMContentLoaded", boot);
