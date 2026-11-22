// Light/dark theme for the adapter report. Colors live in CSS custom
// properties (inspect_adapters.css); canvas drawing reads them from here.
// Shares $ and render from inspect_adapters.js.
const THEME_KEY = "storythread-heatmap-theme";
const CANVAS_VARS = ["floor", "floor-tint", "grid", "label", "tick", "group",
  "caption", "halo", "edge", "outline", "outline-halo", "glow", "blur",
  "glow-selected", "blur-selected", "face-shade"];
let themeCache = null;

// Returns true while the dark theme is active.
function isDark() {
  return document.documentElement.dataset.theme === "dark";
}

// Returns the canvas colors and glow settings of the active theme, with
// numeric settings (glow, blur, faceShade) as numbers.
function theme() {
  if (!themeCache) {
    const css = getComputedStyle(document.documentElement);
    themeCache = {};
    for (const name of CANVAS_VARS) {
      const raw = css.getPropertyValue(`--${name}`).trim();
      const key = name.replace(/-(\w)/g, (_, c) => c.toUpperCase());
      themeCache[key] = isNaN(raw) ? raw : Number(raw);
    }
  }
  return themeCache;
}

// Returns the stored theme choice, or null when the user has not made one.
function savedTheme() {
  try {
    return localStorage.getItem(THEME_KEY);
  } catch {
    return null;  // Storage can be blocked (private mode, file origins).
  }
}

// Applies "light" or "dark" and updates the toggle label. Callers redraw.
function setTheme(name) {
  document.documentElement.dataset.theme = name;
  themeCache = null;
  $("theme").textContent = name === "dark" ? "Light mode" : "Dark mode";
}

// Picks the stored or system theme, then wires the toggle button. The page
// follows the system setting until the user presses the button.
function initTheme() {
  const query = window.matchMedia("(prefers-color-scheme: dark)");
  const system = () => (query.matches ? "dark" : "light");
  let chosen = savedTheme();
  setTheme(chosen || system());
  query.addEventListener("change", () => {
    if (chosen) return;
    setTheme(system());
    render();
  });
  $("theme").addEventListener("click", () => {
    chosen = isDark() ? "light" : "dark";
    try {
      localStorage.setItem(THEME_KEY, chosen);
    } catch {
      // Not saved; the choice still applies to this page view.
    }
    setTheme(chosen);
    render();
  });
}
