// Timeline: lanes, cutoff line, prediction chart, zoom with the browse bar, and mouse gestures.
import { DAY, config, displayName, eventTypeOf, models, riskLabel, shortName, stageNoun, texts } from "../config.js";
import { focusEvent, loadTrajectory, selectNoEvent, setCutoff, setEventType, setKnowledge } from "../controller.js";
import { escapeHtml, percent, plural, timeLabel } from "../format.js";
import { missingShownClass, probability } from "../metrics.js";
import { atEnd, availableEvents, noneAtEnd, cutoffLabel, dayCutoff, endCutoff, eventById, stageCutoff, state } from "../state.js";

const SNAP_PIXELS = 9;
const DRAG_THRESHOLD = 5;
const CHART_PADDING = 6;
// More lanes than this scroll, so the chart and the axis stay in view.
const MAX_VISIBLE_LANES = 10;
// The box-selection menu names at most this many lanes.
const MAX_NAMED_LANES = 3;
// Enough horizontal room for a complete "Day 00" tick without overlap.
const MIN_DAY_TICK_WIDTH = 56;
// Half the width of a day label ("Day 71"): labels keep this far from the ends of the axis
const DAY_LABEL_HALF = 31;
// Zoom: each step halves or doubles the time shown, down to two hours.
const ZOOM_STEP = 2;
const MIN_SPAN = 2 * 3600;
// Marks closer than this many pixels overlap: the browse bar then suggests zooming in.
const CROWDED_PIXELS = 4;
const HOUR = 3600;

const container = document.getElementById("swimlanes");
const detail = document.getElementById("eventDetail");
let drag = null;
let brushIds = null;
let visibleTypes = [];

/** The whole stay, in seconds. */
const total = () => state.stayDays * DAY;
/** The time window shown: the whole stay unless zoomed in (state.zoom). */
const view = () => state.zoom || { start: 0, end: total() };
const fraction = (seconds) => {
  const { start, end } = view();
  return (seconds - start) / (end - start);
};
const position = (seconds) => `${fraction(seconds) * 100}%`;
const inView = (seconds) => {
  const { start, end } = view();
  return seconds >= start && seconds <= end;
};
// The layer above the lanes and the chart: geometry, cutoff line and box selection live here.
const plot = () => container.querySelector(".plot-over");
// Labels of cutoffs close to the right edge go on the left of the line.
const nearEnd = (cutoff) => fraction(cutoff.seconds) > 0.82;

export function init() {
  container.addEventListener("pointerdown", onPointerDown);
  container.addEventListener("pointermove", onPointerMove);
  container.addEventListener("pointerup", onPointerUp);
  container.addEventListener("pointercancel", onPointerUp);
  container.addEventListener("keydown", onKeyDown);
  container.addEventListener("click", onClick);
  container.addEventListener("dblclick", (event) => {
    if (event.target.closest(".overview")) setZoom(null);
  });
  // Ctrl + scroll zooms around the pointer; a sideways scroll (or Shift + scroll) moves the window
  container.addEventListener("wheel", onWheel, { passive: false });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeBrush();
  });
  let resizeFrame = null;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => state.patient && render());
  });
}


export function render() {
  const events = state.patient.events;
  const hasImages = (type) => type.activities.some((activity) => activity.predictionStage);
  visibleTypes = config.eventTypes.filter((type) => hasImages(type) || events.some((event) => eventTypeOf(event.activity) === type));
  // When the lanes scroll, image lanes go first so prediction points stay in view.
  if (visibleTypes.length > MAX_VISIBLE_LANES) {
    visibleTypes = [...visibleTypes.filter(hasImages), ...visibleTypes.filter((type) => !hasImages(type))];
  }
  const known = new Set(availableEvents().map((event) => event.id));
  const outside = new Set(state.explanation?.out_of_context_event_ids || []);

  const rows = visibleTypes.map((type) => {
    const off = laneIsOff(type.key);
    const action = off ? "add these events back" : "remove these events";
    const toggle = `<button class="lane-toggle ${off ? "off" : ""}" data-lane="${type.key}" type="button" aria-pressed="${!off}"
      title="${escapeHtml(type.label)}: click to ${action}"><i style="background: var(--lane-${type.key})"></i><span>${escapeHtml(type.label)}</span></button>`;
    const marks = events.filter((event) => eventTypeOf(event.activity) === type && inView(event.offset_seconds)).map((event) => {
      const off = known.has(event.id) && !state.enabled.has(event.id) ? "off" : "";
      const note = outside.has(event.id) ? ". Out of context: outside the 512 tokens the text models read" : "";
      const title = escapeHtml(`${displayName(event)}, ${timeLabel(event.offset_seconds)}${note}`);
      if (event.prediction_stage) {
        const selected = state.cutoff?.kind === "stage" && state.cutoff.stageId === event.id ? "selected" : "";
        return `<button class="mark image-event ${selected} ${off}" style="left: ${position(event.offset_seconds)}; --c: var(--lane-${type.key})"
          data-stage="${escapeHtml(event.id)}" type="button" title="${title}. Click to predict here.">${escapeHtml(shortName(event) || "IMG")}</button>`;
      }
      const focused = state.focused === event.id ? "focused" : "";
      return `<button class="mark ${off} ${focused}" style="left: ${position(event.offset_seconds)}; --c: var(--lane-${type.key})"
        data-event="${escapeHtml(event.id)}" type="button" title="${title}"></button>`;
    }).join("");
    return `${toggle}<div class="lane">${marks}</div>`;
  }).join("");

  const scrolling = visibleTypes.length > MAX_VISIBLE_LANES;
  const hint = scrolling
    ? `<p class="lanes-hint">Showing ${MAX_VISIBLE_LANES} of ${visibleTypes.length} lanes at a time; scroll the lanes to see the rest.</p>`
    : "";
  const chartLabel = `<span class="chart-label">${escapeHtml(riskLabel)}${models.map((model) =>
    `<small><i class="swatch" style="background: var(--model-${model.id})"></i>${escapeHtml(model.name)}</small>`).join("")}</span>`;

  const scrollTop = container.querySelector(".lane-scroll")?.scrollTop || 0;
  container.innerHTML = `<div class="plot-under">${dayBands()}</div>
    <div class="lane-scroll ${scrolling ? "scrolling" : ""}" style="--visible-lanes: ${MAX_VISIBLE_LANES}">${rows}</div>
    ${hint}${chartLabel}<div class="chart">${chart()}</div>
    <span class="axis-side">
      <button class="all-events" data-all-events="on" type="button" ${atEnd() ? "disabled" : ""}
        title="Move the cutoff to the last event and give the models every event of the case">Select all</button>
      <button class="all-events" data-all-events="off" type="button" ${noneAtEnd() ? "disabled" : ""}
        title="Move the cutoff to the last event and give the models no event: pick events back from the timeline or the list">Deselect all</button></span>
    <div class="axis">${axis()}</div>
    ${browseBar()}
    <div class="plot-over">${cutoffLine()}</div>`;
  container.querySelector(".lane-scroll").scrollTop = scrollTop;
}

function dayBands() {
  return Array.from({ length: state.stayDays }, (_, index) =>
    (index % 2 ? `<div class="day-band" style="left: ${position(index * DAY)}; width: ${(DAY / (view().end - view().start)) * 100}%"></div>` : "")
    + `<div class="day-line" style="left: ${position(index * DAY)}"></div>`).join("");
}

function axis() {
  const { start, end } = view();
  const plotWidth = Math.max(320, container.clientWidth - 148);
  const shownDays = (end - start) / DAY;
  const step = Math.max(1, Math.ceil((shownDays * MIN_DAY_TICK_WIDTH) / plotWidth));
  const activeDay = state.cutoff?.kind === "day" ? state.cutoff.day : null;
  const visibleDay = (day) => (day - 0.5) * DAY >= start && (day - 0.5) * DAY <= end;
  const firstDay = Math.max(1, Math.ceil(start / DAY + 0.5));
  const lastDay = Math.min(state.stayDays, Math.floor(end / DAY + 0.5));
  const anchors = [firstDay, lastDay, activeDay].filter((day) => day && visibleDay(day));
  const shown = new Set(anchors);
  for (let day = firstDay; day <= lastDay; day += step) {
    if (!anchors.some((anchor) => anchor !== day && Math.abs(anchor - day) < step)) shown.add(day);
  }
  // A label near an edge is moved inside the axis (CSS clamp); one that would then cover its neighbour is dropped
  const centre = (day) => Math.min(plotWidth - DAY_LABEL_HALF, Math.max(DAY_LABEL_HALF, fraction((day - 0.5) * DAY) * plotWidth));
  const kept = [];
  for (const day of [...shown].filter(visibleDay).sort((a, b) => a - b)) {
    const previous = kept[kept.length - 1];
    if (previous === undefined || centre(day) - centre(previous) >= 2 * DAY_LABEL_HALF - 2) kept.push(day);
    else if (day === activeDay || (day === lastDay && previous !== activeDay)) kept[kept.length - 1] = day;
  }
  const days = kept.map((day) => {
    const active = state.cutoff?.kind === "day" && state.cutoff.day === day;
    return `<button class="day-button ${active ? "active" : ""}" style="left: clamp(${DAY_LABEL_HALF}px, ${position((day - 0.5) * DAY)}, calc(100% - ${DAY_LABEL_HALF}px))" data-day="${day}" type="button"
      aria-pressed="${active}" title="Predict with everything known by the end of ${texts.day} ${day}">${texts.day} ${day}</button>`;
  }).join("");
  return days + hourTicks(shownDays, plotWidth);
}

/** When a few days or less are shown, hour labels between the day buttons. */
function hourTicks(shownDays, plotWidth) {
  if (shownDays > 3) return "";
  const { start, end } = view();
  const pixelsPerHour = plotWidth / ((end - start) / HOUR);
  const every = [1, 2, 3, 6, 12].find((hours) => hours * pixelsPerHour >= 64) || 12;
  const labels = [];
  for (let at = Math.ceil(start / (every * HOUR)) * every * HOUR; at <= end; at += every * HOUR) {
    const hour = Math.round((at % DAY) / HOUR);
    // Midday sits under the day button; midnight is the day line
    if (hour === 0 || hour === 12) continue;
    labels.push(`<span class="hour-label" style="left: ${position(at)}">${String(hour).padStart(2, "0")}:00</span>`);
  }
  return labels.join("");
}

function cutoffLine() {
  if (!state.cutoff) return "";
  const share = Math.min(1, Math.max(0, fraction(state.cutoff.seconds)));
  const future = `<div class="future" style="left: ${share * 100}%"></div>`;
  if (!inView(state.cutoff.seconds)) return future;  // outside the window: only the shade of what the models do not see
  const at = position(state.cutoff.seconds);
  const label = escapeHtml(cutoffLabel(state.cutoff));
  return `${future}
    <div class="cutoff-line ${nearEnd(state.cutoff) ? "near-end" : ""}" style="left: ${at}" tabindex="0" role="slider" aria-label="Prediction cutoff" aria-valuetext="${label}"><span>${label}</span></div>`;
}

/** The probability of the shown class for every model at every cutoff, to the end of the stay. */
function chart() {
  if (state.trajectoryLoading) return '<div class="chart-message">Computing prediction lines in the background. You can continue navigating while this runs.</div>';
  const points = state.trajectory;
  if (!points) return '<div class="chart-message">Prognosis trajectory not computed yet <button data-compute-trajectory type="button">Compute</button></div>';
  if (missingShownClass(points)) {
    return `<div class="chart-message">These prediction lines were computed before every class was kept, so ${escapeHtml(riskLabel)} is
      not available. Reset the statistics on the Statistics page to compute them again.</div>`;
  }

  const lines = models.map((model) => {
    const coordinates = points.filter((point) => probability(point, model.id) !== null)
      .map((point) => `${fraction(point.offset_seconds) * 1000},${(1 - probability(point, model.id)) * 100}`);
    return coordinates.length ? `<polyline points="${coordinates.join(" ")}" style="stroke: var(--model-${model.id})"/>` : "";
  }).join("");
  const dots = points.filter((point) => inView(point.offset_seconds)).flatMap((point) =>
    models.filter((model) => probability(point, model.id) !== null).map((model) => {
      const value = probability(point, model.id);
      const where = point.kind === "day" ? `end of ${texts.day} ${point.day}` : `${stageNoun} on ${timeLabel(point.offset_seconds)}`;
      const top = `calc(${CHART_PADDING}px + ${(1 - value) * 100}% - ${(1 - value) * 2 * CHART_PADDING}px)`;
      return `<i class="dot ${point.kind === "stage" ? "image-event" : ""}" style="left: ${position(point.offset_seconds)}; top: ${top}; background: var(--model-${model.id})"
        title="${escapeHtml(`${model.name} at the ${where}: ${riskLabel} ${percent(value)}`)}"></i>`;
    })).join("");
  return `<svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true"><line class="threshold" x1="0" x2="1000" y1="50" y2="50"/>${lines}</svg>${dots}
    <span class="chart-axis" style="top: 0">100%</span><span class="chart-axis" style="top: calc(50% - 7px)">50%</span><span class="chart-axis" style="bottom: 0">0%</span>`;
}

/** Whether marks of one lane overlap at the current zoom. */
function crowded() {
  const plotWidth = Math.max(320, container.clientWidth - 148);
  const pixelsPerSecond = plotWidth / (view().end - view().start);
  return visibleTypes.some((type) => {
    const offsets = state.patient.events.filter((event) => eventTypeOf(event.activity) === type && inView(event.offset_seconds))
      .map((event) => event.offset_seconds).sort((a, b) => a - b);
    return offsets.some((offset, index) => index > 0 && offset !== offsets[index - 1]
      && (offset - offsets[index - 1]) * pixelsPerSecond < CROWDED_PIXELS);
  });
}

/**
 * Zoom buttons and the browse bar: every event of the stay as a tick, the shown window as a box
 * that can be dragged, and the cutoff. Clicking the bar moves the window there; double-click shows all.
 */
function browseBar() {
  const all = total();
  const { start, end } = view();
  const zoomed = Boolean(state.zoom);
  const ticks = state.patient.events.map((event) =>
    `<i style="left: ${(event.offset_seconds / all) * 100}%; background: var(--lane-${eventTypeOf(event.activity).key})"></i>`).join("");
  const cutoff = state.cutoff ? `<b class="overview-cutoff" style="left: ${(state.cutoff.seconds / all) * 100}%"></b>` : "";
  const note = !zoomed && crowded() ? '<span class="overview-note">Some events overlap: zoom in with + or Ctrl + scroll</span>' : "";
  return `<span class="zoom-tools">
      <button data-zoom="in" type="button" title="Zoom in (or Ctrl + scroll on the timeline)">+</button>
      <button data-zoom="out" type="button" ${zoomed ? "" : "disabled"} title="Zoom out">-</button>
      <button data-zoom="fit" type="button" ${zoomed ? "" : "disabled"} title="Show the whole case">All</button>
    </span>
    <div class="overview ${zoomed ? "zoomed" : ""}" title="${zoomed ? "Drag the box to browse the case; double-click to show all" : "The whole case; zoom in to browse it"}">
      ${ticks}${cutoff}<div class="overview-window" style="left: ${(start / all) * 100}%; width: ${((end - start) / all) * 100}%"></div>${note}
    </div>`;
}

/** Show the window [start, end] (null: the whole stay), kept inside the stay and at least MIN_SPAN long. */
function setZoom(window) {
  const all = total();
  if (!window || window.end - window.start >= all) {
    state.zoom = null;
  } else {
    const span = Math.max(Math.min(MIN_SPAN, all), window.end - window.start);
    const start = Math.min(Math.max(0, window.start), all - span);
    state.zoom = { start, end: start + span };
  }
  closeBrush();
  render();
}

/** Zoom by a factor around a time (default: the middle of the window). */
function zoomBy(factor, around) {
  const { start, end } = view();
  const centre = around ?? (start + end) / 2;
  const share = (centre - start) / (end - start);
  const span = (end - start) / factor;
  setZoom({ start: centre - share * span, end: centre - share * span + span });
}

/** Move the window so that a time is in view (e.g. after Select all). */
export function reveal(seconds) {
  if (!state.zoom || inView(seconds)) return;
  const span = state.zoom.end - state.zoom.start;
  setZoom({ start: seconds - span * 0.8, end: seconds + span * 0.2 });
}

function onWheel(event) {
  if (!state.patient || !event.target.closest(".lane, .chart, .axis, .overview")) return;
  if (event.ctrlKey || event.metaKey) {
    event.preventDefault();
    const { seconds } = secondsAt(event.clientX);
    zoomBy(event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP, seconds);
  } else if (state.zoom && (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY))) {
    event.preventDefault();
    const { start, end } = view();
    const delta = ((event.shiftKey ? event.deltaY : event.deltaX) / Math.max(1, plot().getBoundingClientRect().width)) * (end - start);
    setZoom({ start: start + delta, end: end + delta });
  }
}

export function describeEvent(eventId) {
  const event = eventById(eventId);
  detail.innerHTML = `<strong>${escapeHtml(displayName(event))}</strong>, ${escapeHtml(timeLabel(event.offset_seconds))}: ${escapeHtml(event.model_text.trim())}`;
}


function secondsAt(clientX) {
  const rect = plot().getBoundingClientRect();
  const share = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  const { start, end } = view();
  return { seconds: start + share * (end - start), pixelsPerSecond: rect.width / (end - start) };
}

/** A time from a position on the browse bar, which always spans the whole stay. */
function overviewSeconds(clientX) {
  const rect = container.querySelector(".overview").getBoundingClientRect();
  return Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)) * total();
}

/** Snap to a nearby image event, then to a nearby end of day, else keep the time. */
function snappedCutoff(clientX) {
  const { seconds, pixelsPerSecond } = secondsAt(clientX);
  const near = (target) => Math.abs(target - seconds) * pixelsPerSecond <= SNAP_PIXELS;
  const stage = state.patient.events.filter((event) => event.prediction_stage)
    .sort((a, b) => Math.abs(a.offset_seconds - seconds) - Math.abs(b.offset_seconds - seconds))[0];
  if (stage && near(stage.offset_seconds)) return stageCutoff(stage);
  const day = Math.round(seconds / DAY);
  if (day >= 1 && day <= state.stayDays && near(day * DAY)) return dayCutoff(day);
  return { kind: "free", seconds: Math.round(seconds / 60) * 60 };
}


function onPointerDown(event) {
  if (!state.patient || event.button !== 0) return;
  const overview = event.target.closest(".overview");
  if (overview) {
    // Browse: drag the window box, or move it to where the bar is pressed and drag from there
    if (!state.zoom) return;
    event.preventDefault();
    const { start, end } = view();
    const at = overviewSeconds(event.clientX);
    const grab = at >= start && at <= end ? at - start : (end - start) / 2;
    if (!(at >= start && at <= end)) setZoom({ start: at - grab, end: at - grab + (end - start) });
    drag = { pointerId: event.pointerId, mode: "browse", grab };
    container.setPointerCapture(event.pointerId);
    return;
  }
  const onLine = event.target.closest(".cutoff-line");
  if (!onLine && (event.target.closest("button, .chart-message, .brush-menu") || !event.target.closest(".lane, .chart"))) return;
  event.preventDefault();
  closeBrush();
  drag = { pointerId: event.pointerId, mode: onLine ? "cutoff" : "pending", x: event.clientX, y: event.clientY, cutoff: null };
  container.setPointerCapture(event.pointerId);
  if (onLine) container.classList.add("dragging");
}

function onPointerMove(event) {
  if (!drag || drag.pointerId !== event.pointerId) return;
  if (drag.mode === "browse") {
    const { start, end } = view();
    const from = overviewSeconds(event.clientX) - drag.grab;
    setZoom({ start: from, end: from + (end - start) });
    return;
  }
  if (drag.mode === "pending" && Math.hypot(event.clientX - drag.x, event.clientY - drag.y) > DRAG_THRESHOLD) drag.mode = "brush";
  if (drag.mode === "cutoff") {
    drag.cutoff = snappedCutoff(event.clientX);
    previewCutoff(drag.cutoff);
  } else if (drag.mode === "brush") {
    drawBrush(drag.x, drag.y, event.clientX, event.clientY);
  }
}

function onPointerUp(event) {
  if (!drag || drag.pointerId !== event.pointerId) return;
  const finished = drag;
  drag = null;
  container.releasePointerCapture(event.pointerId);
  container.classList.remove("dragging");
  if (finished.mode === "browse") return;
  if (event.type === "pointercancel") {
    closeBrush();
    render();
  } else if (finished.mode === "cutoff") {
    if (finished.cutoff) setCutoff(finished.cutoff);
    else render();
  } else if (finished.mode === "brush") {
    openBrushMenu(drawBrush(finished.x, finished.y, event.clientX, event.clientY));
  } else {
    setCutoff(snappedCutoff(event.clientX));
  }
}

function previewCutoff(cutoff) {
  const line = container.querySelector(".cutoff-line");
  if (!line) return;
  line.style.left = position(cutoff.seconds);
  line.classList.toggle("near-end", nearEnd(cutoff));
  container.querySelector(".future").style.left = position(cutoff.seconds);
  line.querySelector("span").textContent = cutoffLabel(cutoff);
}

/** Arrow keys on the focused cutoff line step to the previous or next event. */
function onKeyDown(event) {
  if (!event.target.closest(".cutoff-line") || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault();
  const offsets = [...new Set(state.patient.events.map((item) => item.offset_seconds))].sort((a, b) => a - b);
  const current = state.cutoff.seconds;
  const next = event.key === "ArrowRight"
    ? offsets.find((value) => value > current)
    : offsets.reverse().find((value) => value < current);
  if (next === undefined) return;
  setCutoff({ kind: "free", seconds: next });
  reveal(next);
  container.querySelector(".cutoff-line")?.focus();
}

/** A lane is off when it has known events and none of them is given to the models. */
function laneIsOff(typeKey) {
  const known = availableEvents().filter((event) => eventTypeOf(event.activity).key === typeKey);
  return known.length > 0 && known.every((event) => !state.enabled.has(event.id));
}

function onClick(event) {
  if (event.target.closest("[data-compute-trajectory]")) return loadTrajectory(true);
  const all = event.target.closest("[data-all-events]");
  if (all) {
    const end = endCutoff();
    if (all.dataset.allEvents === "on") setCutoff(end);
    else selectNoEvent();
    return reveal(end.seconds);
  }
  const zoom = event.target.closest("[data-zoom]");
  if (zoom) {
    const action = zoom.dataset.zoom;
    // Zooming in centres on the cutoff when it is in view, so the line stays on screen
    const around = state.cutoff && inView(state.cutoff.seconds) ? state.cutoff.seconds : undefined;
    if (action === "in") return zoomBy(ZOOM_STEP, around);
    if (action === "out") return zoomBy(1 / ZOOM_STEP, around);
    return setZoom(null);
  }
  const lane = event.target.closest("[data-lane]");
  if (lane) return setEventType(lane.dataset.lane, laneIsOff(lane.dataset.lane));
  const day = event.target.closest("[data-day]");
  if (day) return setCutoff(dayCutoff(Number(day.dataset.day)));
  const stage = event.target.closest("[data-stage]");
  if (stage) return setCutoff(stageCutoff(eventById(stage.dataset.stage)));
  const mark = event.target.closest("[data-event]");
  if (mark) return focusEvent(mark.dataset.event);
}


function drawBrush(x1, y1, x2, y2) {
  const area = plot().getBoundingClientRect();
  const box = {
    left: Math.max(area.left, Math.min(x1, x2)),
    right: Math.min(area.right, Math.max(x1, x2)),
    top: Math.max(area.top, Math.min(y1, y2)),
    bottom: Math.min(area.bottom, Math.max(y1, y2)),
  };
  let brush = plot().querySelector(".brush");
  if (!brush) {
    brush = document.createElement("div");
    brush.className = "brush";
    plot().appendChild(brush);
  }
  Object.assign(brush.style, {
    left: `${box.left - area.left}px`, top: `${box.top - area.top}px`,
    width: `${Math.max(0, box.right - box.left)}px`, height: `${Math.max(0, box.bottom - box.top)}px`,
  });
  return box;
}

/** Events inside the box: its time range, on the lanes it covers (all lanes if none). */
function eventsInBox(box) {
  const area = plot().getBoundingClientRect();
  const { start, end } = view();
  const from = start + ((box.left - area.left) / area.width) * (end - start);
  const to = start + ((box.right - area.left) / area.width) * (end - start);
  // Only lanes scrolled into view can be covered by the box.
  const lanes = container.querySelector(".lane-scroll").getBoundingClientRect();
  const covered = [...container.querySelectorAll(".lane")]
    .map((lane, index) => ({ rect: lane.getBoundingClientRect(), type: visibleTypes[index] }))
    .filter(({ rect }) => rect.bottom > Math.max(box.top, lanes.top) && rect.top < Math.min(box.bottom, lanes.bottom))
    .map(({ type }) => type);
  const types = covered.length ? covered : visibleTypes;
  const selected = state.patient.events.filter((event) =>
    event.offset_seconds >= from && event.offset_seconds <= to && types.includes(eventTypeOf(event.activity)));
  return { selected, types: covered.length && covered.length < visibleTypes.length ? covered : null };
}

function openBrushMenu(box) {
  const { selected, types } = eventsInBox(box);
  if (!selected.length) return closeBrush();
  brushIds = new Set(selected.map((event) => event.id));
  const range = `${timeLabel(selected[0].offset_seconds)} to ${timeLabel(selected[selected.length - 1].offset_seconds)}`;
  const lanes = !types ? "all lanes"
    : types.length > MAX_NAMED_LANES ? plural(types.length, "lane") : types.map((type) => type.label).join(", ");
  const area = plot().getBoundingClientRect();
  const menu = document.createElement("div");
  menu.className = "brush-menu";
  menu.innerHTML = `<strong>${plural(selected.length, "event")}</strong><span>${escapeHtml(range)}, ${escapeHtml(lanes)}</span>
    <button data-brush="only" type="button">Use only these</button>
    <button data-brush="exclude" type="button">Exclude</button>
    <button data-brush="include" type="button">Add back</button>
    <button data-brush="cancel" type="button">Cancel</button>`;
  Object.assign(menu.style, {
    left: `${Math.max(0, Math.min(box.left - area.left, area.width - 480))}px`,
    top: `${box.bottom - area.top + 6}px`,
  });
  menu.addEventListener("click", (event) => {
    const button = event.target.closest("[data-brush]");
    if (button) applyBrush(button.dataset.brush);
  });
  plot().appendChild(menu);
}

export function closeBrush() {
  brushIds = null;
  container.querySelectorAll(".brush, .brush-menu").forEach((node) => node.remove());
}

/** Use only, exclude, or add back the selected events. */
function applyBrush(action) {
  const ids = brushIds;
  closeBrush();
  if (action === "cancel" || !ids?.size) return;
  const latest = Math.max(...state.patient.events.filter((event) => ids.has(event.id)).map((event) => event.offset_seconds));
  const lastEvent = state.patient.events[state.patient.events.length - 1].offset_seconds;
  const before = state.cutoff ? new Set(state.enabled) : null;

  let cutoff = state.cutoff || { kind: "free", seconds: action === "exclude" ? lastEvent : latest };
  if (action !== "exclude" && cutoff.seconds < latest) cutoff = { kind: "free", seconds: latest };
  const known = availableEvents(cutoff).map((event) => event.id);
  const keep = {
    only: (id) => ids.has(id),
    exclude: (id) => (before ? before.has(id) : true) && !ids.has(id),
    include: (id) => (before ? before.has(id) : false) || ids.has(id),
  }[action];
  setKnowledge(cutoff, known.filter(keep));
}
