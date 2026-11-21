// The bar between the timeline and the panels. Dragging it down makes the prediction chart taller;
// dragging it up shrinks the chart, then the timeline area, which then scrolls. The size is kept
// in the browser; a double-click restores the default.
const STORAGE_KEY = "explorer.split";
const DEFAULT_CHART = 84;
const MIN_CHART = 48;
const MIN_PATHWAY = 140;
const MIN_PANELS = 260;  // as --min-panels in style.css
const KEY_STEP = 24;

const bar = document.getElementById("splitter");
const pathway = document.querySelector(".pathway");
const workspace = document.getElementById("patientView");
let size = { chart: DEFAULT_CHART, clip: null };  // clip: the height of a shrunk timeline area, in pixels
let drag = null;

function apply() {
  document.documentElement.style.setProperty("--chart-height", `${size.chart}px`);
  pathway.style.height = size.clip ? `${size.clip}px` : "";
  pathway.classList.toggle("clipped", Boolean(size.clip));
  bar.setAttribute("aria-valuenow", String(Math.round(pathway.getBoundingClientRect().height)));
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(size));
  } catch (_) {
    // Private windows may refuse storage; the size then lasts until a reload
  }
}

function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (saved && Number.isFinite(saved.chart)) size = { chart: Math.max(MIN_CHART, saved.chart), clip: saved.clip || null };
  } catch (_) {
    size = { chart: DEFAULT_CHART, clip: null };
  }
}

/** The timeline height wanted, turned into a chart height or a clipped timeline area. */
function resizeTo(start, wanted) {
  const room = workspace.getBoundingClientRect().height - bar.getBoundingClientRect().height - MIN_PANELS;
  // The case view scrolls when the window is short, so growing never clips the timeline below its natural height
  const height = Math.min(Math.max(wanted, MIN_PATHWAY), Math.max(MIN_PATHWAY, room, start.naturalAtMinimum));
  if (height >= start.naturalAtMinimum) size = { chart: MIN_CHART + (height - start.naturalAtMinimum), clip: null };
  else size = { chart: MIN_CHART, clip: height };
  apply();
}

function startSize() {
  const height = pathway.getBoundingClientRect().height;
  // The timeline height with the smallest chart and no clipping
  const naturalAtMinimum = size.clip ? pathway.scrollHeight - (size.chart - MIN_CHART) : height - (size.chart - MIN_CHART);
  return { height, naturalAtMinimum };
}

export function init() {
  restore();
  apply();
  bar.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    drag = { pointerId: event.pointerId, y: event.clientY, ...startSize() };
    bar.setPointerCapture(event.pointerId);
    document.body.classList.add("resizing");
  });
  bar.addEventListener("pointermove", (event) => {
    if (drag?.pointerId === event.pointerId) resizeTo(drag, drag.height + event.clientY - drag.y);
  });
  const end = (event) => {
    if (drag?.pointerId !== event.pointerId) return;
    drag = null;
    document.body.classList.remove("resizing");
    save();
  };
  bar.addEventListener("pointerup", end);
  bar.addEventListener("pointercancel", end);
  bar.addEventListener("dblclick", () => {
    size = { chart: DEFAULT_CHART, clip: null };
    apply();
    save();
  });
  bar.addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const start = startSize();
    resizeTo(start, start.height + (event.key === "ArrowDown" ? KEY_STEP : -KEY_STEP));
    save();
  });
}
