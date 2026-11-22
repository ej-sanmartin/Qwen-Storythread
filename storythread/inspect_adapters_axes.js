// Floor, group tints, axis labels, and group brackets for the 3D scenes.
// Calls rotate, NX, NZ, ZSTEP, and GAP from inspect_adapters_3d.js at draw
// time. Groups come from the glossary, so they appear only when the
// architecture is known; otherwise columns are plain exact names.
const LABEL_FONT = "11px system-ui";
const GROUP_FONT = "600 11px system-ui";
const GROUP_BAND = 34;  // pixels reserved under the labels for brackets
const GROUP_RUNS = groupRuns();

// Returns [{group, from, to}] for runs of adjacent columns in one group.
function groupRuns() {
  const runs = [];
  REPORT.columns.forEach((c, i) => {
    const about = describe(c.id);
    if (!about) return;
    const last = runs[runs.length - 1];
    if (last && last.group === about.group && last.to === i - 1) last.to = i;
    else runs.push({group: about.group, from: i, to: i});
  });
  return runs;
}

// Returns the pixels the vertical projection labels need under the floor.
function labelRoom(ctx) {
  ctx.font = LABEL_FONT;
  const widths = REPORT.columns.map((c) => ctx.measureText(c.label).width);
  return Math.min(150, Math.max(...widths));
}

// Returns the pixels needed under the floor for labels and group brackets.
function bottomRoom(ctx) {
  return 22 + labelRoom(ctx) + (GROUP_RUNS.length ? GROUP_BAND : 0);
}

// Traces the floor strip covering columns from x0 to x1 (world units).
function traceStrip(ctx, at, x0, x1) {
  const depth = NZ * ZSTEP;
  ctx.beginPath();
  [[x0, 0], [x1, 0], [x1, depth], [x0, depth]].forEach(([x, z], k) => {
    const [px, py] = at(x, 0, z);
    if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
  });
  ctx.closePath();
}

// Draws the floor, a faint tint under every other group, and cell lines.
function drawFloor(ctx, at) {
  const depth = NZ * ZSTEP;
  traceStrip(ctx, at, 0, NX);
  ctx.fillStyle = "#f3f4f6";
  ctx.fill();
  GROUP_RUNS.forEach((run, k) => {
    if (k % 2) return;
    traceStrip(ctx, at, run.from, run.to + 1);
    ctx.fillStyle = "rgba(71,98,140,.07)";
    ctx.fill();
  });
  ctx.strokeStyle = "#e1e4e8";
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

// Draws text with a white halo so bars behind it never hide it.
function haloText(ctx, text, x, y) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(255,255,255,.92)";
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}

// Draws vertical projection names under the near edge, with a bracket and
// title under each known group.
function drawProjectionLabels(ctx, at, room) {
  const depth = NZ * ZSTEP;
  const nearZ = rotate(NX / 2, 0, 0).d < rotate(NX / 2, 0, depth).d ? 0 : depth;
  ctx.fillStyle = "#374151";
  ctx.font = LABEL_FONT;
  ctx.textAlign = "right";
  const anchors = REPORT.columns.map((c, i) => at(i + 0.5, 0, nearZ));
  REPORT.columns.forEach((c, i) => {
    const [x, y] = anchors[i];
    ctx.save();
    ctx.translate(x + 4, y + 8);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(c.label, 0, 0);
    ctx.restore();
  });
  // One shared baseline under the lowest label, so group titles line up.
  const base = Math.max(...anchors.map((a) => a[1])) + 8 + room + 12;
  ctx.strokeStyle = "#9ca3af";
  ctx.lineWidth = 1;
  ctx.font = GROUP_FONT;
  ctx.textAlign = "center";
  for (const run of GROUP_RUNS) {
    const [x0] = at(run.from + GAP, 0, nearZ);
    const [x1] = at(run.to + 1 - GAP, 0, nearZ);
    ctx.beginPath();
    ctx.moveTo(x0, base - 4);
    ctx.lineTo(x0, base);
    ctx.lineTo(x1, base);
    ctx.lineTo(x1, base - 4);
    ctx.stroke();
    ctx.fillStyle = "#4b5563";
    ctx.fillText(run.group, (x0 + x1) / 2, base + 15);
  }
}

// Draws block labels beyond the outer side edge of the floor, each with a
// short tick, on whichever side edge is currently at screen left.
function drawBlockLabels(ctx, at) {
  const mid = NZ * ZSTEP / 2;
  const edge = rotate(0, 0, mid).x < rotate(NX, 0, mid).x ? 0 : NX;
  const outward = edge === 0 ? -1 : 1;
  ctx.font = LABEL_FONT;
  ctx.strokeStyle = "#9ca3af";
  ctx.lineWidth = 1;
  REPORT.blocks.forEach((b, j) => {
    const z = (j + 0.5) * ZSTEP;
    const [x0, y0] = at(edge, 0, z);
    const [x1, y1] = at(edge + outward * 0.45, 0, z);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.fillStyle = "#374151";
    ctx.textAlign = x1 < x0 ? "right" : "left";
    haloText(ctx, b === null ? "other" : `block ${b}`,
             x1 + (x1 < x0 ? -4 : 4), y1 + 4);
  });
}

// Draws every axis label for a scene; room is from labelRoom.
function drawLabels(ctx, at, room) {
  drawProjectionLabels(ctx, at, room);
  drawBlockLabels(ctx, at);
}
