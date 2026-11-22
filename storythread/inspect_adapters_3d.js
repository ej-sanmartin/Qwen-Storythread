// 3D bar-chart views for the adapter report. Shares globals with
// inspect_adapters.js (REPORT, state, valueAt, ramp, signed, el, render)
// and inspect_adapters_axes.js (floor and labels).
const HEIGHT = 2;       // world height of a full-scale bar
const ZSTEP = 1.8;      // world depth of one block row
const GAP = 0.14;       // empty margin inside each grid cell
const MIN_H = 0.03;     // keeps zero-height bars visible
// Orthographic camera: no perspective, so equal bars stay equal size.
// A higher pitch looks down more, which exposes the back rows.
const HOME = {yaw: 0.5, pitch: 0.9};
const camera = {...HOME};
const scenes = [];      // {canvas, bars, caption, low, hits, links, ctx, dpr}
const NX = REPORT.columns.length;
const NZ = REPORT.blocks.length;
const COLUMN_INDEX = new Map(REPORT.columns.map((c, i) => [c.id, i]));
const COLUMN_LABEL = new Map(REPORT.columns.map((c) => [c.id, c.label]));
// Outward normal, corner picks (0 = min, 1 = max per axis), lightness shift.
const FACES = [
  {n: [0, 1, 0], c: [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]], dl: 0},
  {n: [0, -1, 0], c: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dl: -6},
  {n: [1, 0, 0], c: [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]], dl: -9},
  {n: [-1, 0, 0], c: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dl: -9},
  {n: [0, 0, 1], c: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dl: -15},
  {n: [0, 0, -1], c: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dl: -15},
];

// Returns recent vs earlier as a percent of earlier, or undefined.
function diffPercent(path, step) {
  const [a, b] = REPORT.arms.map((arm) => valueAt(arm, step, path));
  return a === undefined || b === undefined || b === 0
    ? undefined : 100 * (a - b) / b;
}

// Largest absolute percent gap over all steps and modules (at least 1).
const DIFF_MAX = Math.ceil(Math.max(1, ...REPORT.steps.flatMap((step) =>
  Object.keys(REPORT.modules).map((p) => Math.abs(diffPercent(p, step) || 0)))));

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// Returns the view-space point {x, y, d} for a world point; larger d is
// farther from the camera.
function rotate(px, py, pz) {
  const dx = px - NX / 2, dz = pz - NZ * ZSTEP / 2;
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const z1 = dx * sy + dz * cy;
  return {x: dx * cy - dz * sy, y: py * cp + z1 * sp, d: z1 * cp - py * sp};
}

// Returns a function mapping world points to canvas pixels so the floor,
// from height low to HEIGHT, fills a w by h canvas. bottom is the pixel room
// kept for labels under the floor.
function fitter(w, h, low, bottom) {
  const pts = [];
  for (const x of [0, NX]) for (const z of [0, NZ * ZSTEP]) {
    for (const y of [low, HEIGHT]) pts.push(rotate(x, y, z));
  }
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const pad = {left: 70, right: 12, top: 26, bottom};
  const k = Math.min((w - pad.left - pad.right) / (x1 - x0),
                     (h - pad.top - pad.bottom) / (y1 - y0));
  const ox = pad.left + (w - pad.left - pad.right) / 2 - k * (x0 + x1) / 2;
  const oy = pad.top + (h - pad.top - pad.bottom) / 2 + k * (y0 + y1) / 2;
  return (x, y, z) => {
    const p = rotate(x, y, z);
    return [ox + k * p.x, oy - k * p.y];
  };
}

// Returns the bar's glow color at the given alpha, from its own hue so the
// glow reads as light from the bar (blue, or amber in the diff view).
function glowColor(bar, alpha) {
  return `hsl(${bar.hue} ${bar.sat}% 55% / ${alpha})`;
}

// Draws one bar's visible faces and records them for hit testing. Every bar
// glows a little; the selected bar glows more.
function drawBar(ctx, at, bar, scene) {
  const t = theme();
  const h = bar.h < 0 ? Math.min(bar.h, -MIN_H) : Math.max(bar.h, MIN_H);
  const axes = [[bar.i + GAP, bar.i + 1 - GAP], [Math.min(0, h), Math.max(0, h)],
                [(bar.j + GAP) * ZSTEP, (bar.j + 1 - GAP) * ZSTEP]];
  const [cy, sy] = [Math.cos(camera.yaw), Math.sin(camera.yaw)];
  const [cp, sp] = [Math.cos(camera.pitch), Math.sin(camera.pitch)];
  for (const face of FACES) {
    const [nx, ny, nz] = face.n;
    if (-ny * sp + (nx * sy + nz * cy) * cp >= 0) continue;
    const path = new Path2D();
    face.c.forEach(([a, b, c], k) => {
      const [x, y] = at(axes[0][a], axes[1][b], axes[2][c]);
      if (k) path.lineTo(x, y); else path.moveTo(x, y);
    });
    path.closePath();
    ctx.fillStyle =
      `hsl(${bar.hue} ${bar.sat}% ${bar.light + face.dl * t.faceShade}%)`;
    ctx.shadowColor = glowColor(bar, bar.selected ? t.glowSelected : t.glow);
    ctx.shadowBlur = (bar.selected ? t.blurSelected : t.blur) * scene.dpr;
    ctx.fill(path);
    ctx.shadowBlur = 0;
    ctx.strokeStyle = t.edge;
    ctx.lineWidth = 0.6;
    ctx.stroke(path);
    scene.hits.push({path, bar});
  }
}

// Outlines the selected bar's faces on top, with a strong glow; the fill is
// left untouched so color and height still show the update size.
function outlineSelected(ctx, scene) {
  const t = theme();
  const faces = scene.hits.filter((hit) => hit.bar.selected);
  if (!faces.length) return;
  ctx.save();
  ctx.lineJoin = "round";
  ctx.shadowColor = glowColor(faces[0].bar, t.glowSelected);
  ctx.shadowBlur = t.blurSelected * scene.dpr;
  for (const [color, width] of [[t.outlineHalo, 5], [t.outline, 2]]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    for (const hit of faces) ctx.stroke(hit.path);
  }
  ctx.restore();
}

// Redraws one scene on its canvas for the current camera.
function drawScene(scene) {
  const {canvas, bars} = scene;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const room = labelRoom(ctx);
  const at = fitter(w, h, scene.low, bottomRoom(ctx));
  drawFloor(ctx, at);
  scene.hits = [];
  scene.ctx = ctx;
  scene.dpr = dpr;
  const far = (b) => rotate(b.i + 0.5, 0, (b.j + 0.5) * ZSTEP).d;
  for (const bar of [...bars].sort((p, q) => far(q) - far(p))) {
    drawBar(ctx, at, bar, scene);
  }
  outlineSelected(ctx, scene);
  drawLabels(ctx, at, room, scene);
  ctx.fillStyle = theme().caption;
  ctx.font = "12px system-ui";
  ctx.textAlign = "left";
  ctx.fillText(scene.caption, 10, 16);
}

// Redraws every scene on screen.
function drawScenes() {
  scenes.forEach(drawScene);
}

// Returns a description such as "block 29 · in_proj_a".
function where(info) {
  const block = info.block === null ? "other" : `block ${info.block}`;
  return `${block} \u00b7 ${COLUMN_LABEL.get(info.column)}`;
}

// Returns bars for every module that measure(path, info) describes.
function gridBars(measure) {
  const bars = [];
  for (const [path, info] of Object.entries(REPORT.modules)) {
    const bar = measure(path, info);
    if (!bar) continue;
    bars.push({path, i: COLUMN_INDEX.get(info.column),
               j: REPORT.blocks.indexOf(info.block),
               selected: path === state.module, ...bar});
  }
  return bars;
}

// Returns a panel holding one 3D scene with a hover readout.
function scenePanel(title, bars, caption, low, tall, note) {
  const canvas = el("canvas", {
    class: tall ? "scene tall" : "scene", role: "img",
    "aria-label": `${title}: 3D bar chart. The Flat grid view is the ` +
                  "keyboard-friendly version."});
  const readout = el("p", {class: "readout"}, HINT);
  const scene = {canvas, bars, caption, low, hits: [], links: []};
  scenes.push(scene);
  attach(scene, readout);
  const notes = note ? [el("p", {class: "absent"}, note)] : [];
  return el("section", {class: "panel"}, el("h2", {}, title), ...notes,
            canvas, readout);
}

// Returns one arm's 3D update-size panel for the selected step.
function sizePanel(arm) {
  const bars = gridBars((path, info) => {
    const v = valueAt(arm, state.step, path);
    if (v === undefined) return null;
    const t = Math.min(1, v / (REPORT.max || 1));
    return {h: t * HEIGHT, ...armTone(arm, t),
            label: `${arm} \u00b7 ${where(info)} \u00b7 RMS ${v.toPrecision(3)}`};
  });
  const note = String(state.step) in REPORT.rms[arm]
    ? "" : `No snapshot for ${arm} at step ${state.step}.`;
  return scenePanel(`${arm} \u00b7 step ${state.step}`, bars,
    "Taller and deeper: larger effective LoRA update RMS.", 0, false, note);
}

// Returns the 3D panel of the first arm minus the second, in percent.
function diffPanel() {
  const [a, b] = REPORT.arms;
  const bars = gridBars((path, info) => {
    const pct = diffPercent(path, state.step);
    if (pct === undefined) return null;
    const t = Math.min(1, Math.abs(pct) / DIFF_MAX);
    const color = pct >= 0 ? armTone(a, t) : armTone(b, t);
    return {h: Math.sign(pct) * t * HEIGHT, ...color,
            label: `${where(info)} \u00b7 ${a} is ${signed(pct)} vs ${b}`};
  });
  const note = bars.length ? "" :
    `Both arms need a snapshot at step ${state.step}.`;
  return scenePanel(`${a} vs ${b} \u00b7 step ${state.step}`, bars,
    `Above the floor: ${a} larger. Below: ${b} larger. Height: % of ${b}.`,
    -HEIGHT, true, note);
}
