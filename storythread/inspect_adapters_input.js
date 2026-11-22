// Pointer input for the 3D scenes: drag to rotate, hover readout, click a
// bar to select it, click an axis label to read its glossary entry. Shares
// scenes, camera, clamp, drawScenes (inspect_adapters_3d.js), goToGlossary
// (detail), and state and render (main).
const YAW_LIMIT = 1.2;  // keeps the side labels on the outside
const HINT = "Drag to rotate. Click a bar for details" +
  (ARCHITECTURE_KNOWN ? ", or a label to read what it means." : ".");

// Returns the pointer position in the canvas's device pixels, the space
// used by isPointInPath while the dpr transform is active.
function devicePoint(scene, event) {
  const rect = scene.canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) * scene.dpr,
          (event.clientY - rect.top) * scene.dpr];
}

// Returns the bar under a pointer event, or undefined.
function pick(scene, event) {
  const [x, y] = devicePoint(scene, event);
  const hit = [...scene.hits].reverse().find(
    (h) => scene.ctx.isPointInPath(h.path, x, y));
  return hit && hit.bar;
}

// Returns the axis-label link under a pointer event, or undefined.
function pickLink(scene, event) {
  const [x, y] = devicePoint(scene, event);
  return scene.links.find((l) => scene.ctx.isPointInPath(l.path, x, y));
}

// Wires drag-to-rotate, hover readout, and click handling on a scene.
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
      const link = pickLink(scene, e);
      const bar = link ? null : pick(scene, e);
      readout.textContent = link ? link.hint : bar ? bar.label : HINT;
      canvas.style.cursor = link || bar ? "pointer" : "grab";
      return;
    }
    if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 4) {
      drag.moved = true;
    }
    if (!drag.moved) return;
    camera.yaw = clamp(camera.yaw + (e.clientX - drag.last[0]) * 0.01,
                       -YAW_LIMIT, YAW_LIMIT);
    camera.pitch = clamp(camera.pitch + (e.clientY - drag.last[1]) * 0.006,
                         0.35, 1.4);
    drag.last = [e.clientX, e.clientY];
    drawScenes();
  });
  canvas.addEventListener("pointerup", (e) => {
    const clicked = drag && !drag.moved;
    drag = null;
    if (!clicked) return;
    const link = pickLink(scene, e);
    if (link) {
      goToGlossary(link.ids);
      return;
    }
    const bar = pick(scene, e);
    if (bar) { state.module = bar.path; render(); }
  });
}
