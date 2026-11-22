// 3D bar-chart views for the adapter report. Shares globals with
// inspect_adapters.js (REPORT, state, valueAt, ramp, signed, el, render).
const HEIGHT = 2.4;     // world height of a full-scale bar
const ZSTEP = 1.9;      // world depth of one block row
const GAP = 0.14;       // empty margin inside each grid cell
const MIN_H = 0.03;     // keeps zero-height bars visible
const HINT = "Drag to rotate. Click a bar for details.";
const HOME = {yaw: 0.35, pitch: 0.6};
const camera = {...HOME};
const scenes = [];      // {canvas, bars, caption, low, hits, ctx, dpr}
const NX = REPORT.columns.length;
const NZ = REPORT.blocks.length;
const COLUMN_INDEX = new Map(REPORT.columns.map((c, i) => [c.id, i]));
const COLUMN_LABEL = new Map(REPORT.columns.map((c) => [c.id, c.label]));
// Outward normal, corner picks (0 = min, 1 = max per axis), lightness shift.
const FACES = [
  {n: [0, 1, 0], c: [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]], dl: 0},
  {n: [0, -1, 0], c: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dl: -8},
  {n: [1, 0, 0], c: [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]], dl: -12},
  {n: [-1, 0, 0], c: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dl: -12},
  {n: [0, 0, 1], c: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dl: -22},
  {n: [0, 0, -1], c: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dl: -22},
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

// Returns a function mapping world points to canvas pixels so the whole
// floor, from height low to HEIGHT, fits a w by h canvas.
function fitter(w, h, low) {
  const pts = [];
  for (const x of [0, NX]) for (const z of [0, NZ * ZSTEP]) {
    for (const y of [low, HEIGHT]) pts.push(rotate(x, y, z));
  }
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const pad = {left: 80, right: 24, top: 34, bottom: 84};  // label room
  const k = Math.min((w - pad.left - pad.right) / (x1 - x0),
                     (h - pad.top - pad.bottom) / (y1 - y0));
  const ox = pad.left + (w - pad.left - pad.right) / 2 - k * (x0 + x1) / 2;
  const oy = pad.top + (h - pad.top - pad.bottom) / 2 + k * (y0 + y1) / 2;
  return (x, y, z) => {
    const p = rotate(x, y, z);
    return [ox + k * p.x, oy - k * p.y];
  };
}

// Draws the floor and its cell lines.
function drawFloor(ctx, at) {
  const depth = NZ * ZSTEP;
  const corners = [[0, 0], [NX, 0], [NX, depth], [0, depth]].map(
    ([x, z]) => at(x, 0, z));
  ctx.beginPath();
  corners.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = "#f4f4f5";
  ctx.fill();
  ctx.strokeStyle = "#e4e4e7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= NX; i++) {
    ctx.moveTo(...at(i, 0, 0));
    ctx.lineTo(...at(i, 0, depth));
  }
  for (let j = 0; j <= NZ; j++) {
    ctx.moveTo(...at(0, 0, j * ZSTEP));
    ctx.lineTo(...at(NX, 0, j * ZSTEP));
  }
  ctx.stroke();
}

// Draws text with a white halo so bars behind it do not hide it.
function haloText(ctx, text, x, y) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(255,255,255,.9)";
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}

// Draws projection names on the near edge and block numbers on the left.
function drawLabels(ctx, at) {
  const depth = NZ * ZSTEP;
  const nearZ = rotate(NX / 2, 0, 0).d < rotate(NX / 2, 0, depth).d ? 0 : depth;
  const labelZ = nearZ === 0 ? -0.4 : depth + 0.4;
  ctx.fillStyle = "#3f3f46";
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  REPORT.columns.forEach((c, i) => {
    const [x, y] = at(i + 0.5, 0, labelZ);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-Math.PI / 4);
    haloText(ctx, c.label, 0, 0);
    ctx.restore();
  });
  const mid = depth / 2;
  const leftX = rotate(0, 0, mid).x < rotate(NX, 0, mid).x ? 0 : NX;
  const labelX = leftX === 0 ? -0.4 : NX + 0.4;
  ctx.textAlign = leftX === 0 ? "right" : "left";
  REPORT.blocks.forEach((b, j) => {
    const [x, y] = at(labelX, 0, (j + 0.5) * ZSTEP);
    haloText(ctx, b === null ? "other" : `block ${b}`, x, y + 4);
  });
}

// Draws one bar's visible faces and records them for hit testing.
function drawBar(ctx, at, bar, scene) {
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
    ctx.fillStyle = `hsl(${bar.hue} ${bar.sat}% ${bar.light + face.dl}%)`;
    ctx.fill(path);
    ctx.strokeStyle = "rgba(0,0,0,.3)";
    ctx.lineWidth = 0.6;
    ctx.stroke(path);
    scene.hits.push({path, bar});
  }
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
  const at = fitter(w, h, scene.low);
  drawFloor(ctx, at);
  scene.hits = [];
  scene.ctx = ctx;
  scene.dpr = dpr;
  const far = (b) => rotate(b.i + 0.5, 0, (b.j + 0.5) * ZSTEP).d;
  for (const bar of [...bars].sort((p, q) => far(q) - far(p))) {
    drawBar(ctx, at, bar, scene);
  }
  ctx.strokeStyle = "#18181b";
  ctx.lineWidth = 2.5;
  for (const hit of scene.hits) if (hit.bar.selected) ctx.stroke(hit.path);
  drawLabels(ctx, at);
  ctx.fillStyle = "#52525b";
  ctx.font = "12px system-ui";
  ctx.textAlign = "left";
  ctx.fillText(scene.caption, 12, 20);
}

// Redraws every scene on screen.
function drawScenes() {
  scenes.forEach(drawScene);
}

// Returns the bar under a pointer event, or undefined.
function pick(scene, event) {
  const rect = scene.canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * scene.dpr;
  const y = (event.clientY - rect.top) * scene.dpr;
  for (let k = scene.hits.length - 1; k >= 0; k--) {
    if (scene.ctx.isPointInPath(scene.hits[k].path, x, y)) {
      return scene.hits[k].bar;
    }
  }
  return undefined;
}

// Wires drag-to-rotate, hover readout, and click-to-select on a scene.
function attach(scene, readout) {
  const canvas = scene.canvas;
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    drag = {x: e.clientX, y: e.clientY, last: [e.clientX, e.clientY],
            moved: false};
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) {
      const bar = pick(scene, e);
      readout.textContent = bar ? bar.label : HINT;
      canvas.style.cursor = bar ? "pointer" : "grab";
      return;
    }
    if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 4) {
      drag.moved = true;
    }
    if (!drag.moved) return;
    camera.yaw += (e.clientX - drag.last[0]) * 0.01;
    camera.pitch = clamp(camera.pitch + (e.clientY - drag.last[1]) * 0.006,
                         0.15, 1.45);
    drag.last = [e.clientX, e.clientY];
    drawScenes();
  });
  canvas.addEventListener("pointerup", (e) => {
    const bar = drag && !drag.moved ? pick(scene, e) : null;
    drag = null;
    if (bar) { state.module = bar.path; render(); }
  });
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
  const scene = {canvas, bars, caption, low, hits: []};
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
    return {h: t * HEIGHT, ...ramp(t),
            label: `${arm} \u00b7 ${where(info)} \u00b7 RMS ${v.toPrecision(3)}`};
  });
  const note = String(state.step) in REPORT.rms[arm]
    ? "" : `No snapshot for ${arm} at step ${state.step}.`;
  return scenePanel(`${arm} \u00b7 step ${state.step}`, bars,
    "Left-right: projection. Front-back: block. Height and color: update RMS.",
    0, false, note);
}

// Returns the 3D panel of the first arm minus the second, in percent.
function diffPanel() {
  const [a, b] = REPORT.arms;
  const bars = gridBars((path, info) => {
    const pct = diffPercent(path, state.step);
    if (pct === undefined) return null;
    const t = Math.min(1, Math.abs(pct) / DIFF_MAX);
    return {h: Math.sign(pct) * t * HEIGHT, hue: pct >= 0 ? 215 : 28,
            sat: 80, light: 92 - 52 * t,
            label: `${where(info)} \u00b7 ${a} is ${signed(pct)} vs ${b}`};
  });
  const note = bars.length ? "" :
    `Both arms need a snapshot at step ${state.step}.`;
  return scenePanel(`${a} vs ${b} \u00b7 step ${state.step}`, bars,
    `Above the floor: ${a} larger. Below: ${b} larger. Height: % of ${b}.`,
    -HEIGHT, true, note);
}
