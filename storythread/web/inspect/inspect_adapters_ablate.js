// Ablation selection for the adapter report. A selection belongs to one arm
// and one checkpoint step, so changing the step or the view shows that
// pair's own set. Shares REPORT, state, valueAt, el, and $ with the main
// script. Inspect mode does not read this.
const PLAN_SCHEMA = 1;
const picked = new Map();  // "arm|step" -> Set of full module paths

// Returns the selection set for an arm at a step (the current step by default).
function chosen(arm, step = state.step) {
  const key = `${arm}|${step}`;
  if (!picked.has(key)) picked.set(key, new Set());
  return picked.get(key);
}

// Returns the module paths in one block that this arm has at the current step.
function blockModules(arm, block) {
  return Object.entries(REPORT.modules)
    .filter(([path, info]) => info.block === block
      && valueAt(arm, state.step, path) !== undefined)
    .map(([path]) => path);
}

// Toggles one module in the current arm and step, then redraws.
function toggleModule(arm, path) {
  const set = chosen(arm);
  if (set.has(path)) set.delete(path);
  else set.add(path);
  render();
}

// Selects every adapted projection in the block, or clears them if all are
// already selected. Modules with no tensor at this step are left alone.
function toggleBlock(arm, block) {
  const paths = blockModules(arm, block);
  if (!paths.length) return;
  const set = chosen(arm);
  const clear = paths.every((path) => set.has(path));
  paths.forEach((path) => (clear ? set.delete(path) : set.add(path)));
  render();
}

// Returns the plan object for an arm's selection at the current step.
function planFor(arm) {
  const saved = REPORT.checkpoints[arm] || {};
  const checkpoint = saved[String(state.step)];
  return {
    schema: PLAN_SCHEMA, arm,
    checkpoint: checkpoint && checkpoint.path,
    sha256: checkpoint && checkpoint.sha256,
    modules: [...chosen(arm)].sort(),
  };
}

// Downloads one arm's plan. The caller has already checked the checkpoint.
function exportPlan(arm) {
  const plan = planFor(arm);
  const blob = new Blob([JSON.stringify(plan, null, 2) + "\n"],
    {type: "application/json"});
  const link = el("a", {
    href: URL.createObjectURL(blob),
    download: `ablation-${arm}-step${state.step}.json`,
  });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// Returns one arm's group: the count, the paths, and an export button when
// withButton is set (more than one arm is selected).
function ablationGroup(arm, withButton) {
  const modules = [...chosen(arm)].sort();
  const saved = REPORT.checkpoints[arm] || {};
  const checkpoint = saved[String(state.step)];
  const noun = modules.length === 1 ? "projection" : "projections";
  const props = {type: "button", onclick: () => exportPlan(arm)};
  if (!checkpoint) props.disabled = "disabled";
  const button = withButton ? [el("button", props, "Export ablation plan")] : [];
  const note = checkpoint ? "" : " This report has no checkpoint file or hash.";
  return el("div", {class: "ablation-arm"},
    el("h3", {},
      `${arm} \u00b7 step ${state.step} \u00b7 ${modules.length} ${noun}`),
    ...button,
    el("p", {class: "hint"}, `Full module paths.${note}`),
    el("ul", {}, ...modules.map((path) => el("li", {}, el("code", {}, path)))));
}

// Fills the selection list for the current step. Hidden in Inspect mode.
function renderAblation() {
  const box = $("ablation");
  box.hidden = state.mode !== "ablate";
  if (box.hidden) return;
  const arms = REPORT.arms.filter((arm) => chosen(arm).size);
  const ready = arms.length === 1 && planFor(arms[0]).checkpoint;
  const props = {id: "export", type: "button"};
  if (!ready) props.disabled = "disabled";
  else props.onclick = () => exportPlan(arms[0]);
  const button = el("button", props, "Export ablation plan");
  const lead = "Both arms, side by side. Click a bar to toggle that "
    + "projection. Click a block label to select or clear the whole block. "
    + "The checkpoint step is the snapshot this plan will copy.";
  const groups = arms.map((arm) => ablationGroup(arm, arms.length !== 1));
  const empty = arms.length ? [] : [el("p", {class: "hint"},
    "No projections selected at this checkpoint.")];
  const many = arms.length > 1 ? [el("p", {class: "hint"},
    "A plan is one arm. Use the button on each arm.")] : [];
  box.replaceChildren(
    el("div", {class: "ablation-bar"}, el("h2", {}, "Ablation selection"), button),
    el("p", {class: "hint"}, lead), ...empty, ...many, ...groups);
}
