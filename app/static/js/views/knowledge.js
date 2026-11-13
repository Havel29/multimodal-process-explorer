// The list of every event of the case, with a checkbox per event and a search over them. Events after the
// cutoff stay listed, greyed out and without a checkbox, so the list does not change length with the cutoff.
import { displayName, eventTypeOf, revealsOutcome, texts } from "../config.js";
import { toggleEvents } from "../controller.js";
import { clock, dayOf, escapeHtml, plural, timeLabel } from "../format.js";
import { availableEvents, state } from "../state.js";
import { divergingBar } from "./bars.js";

const els = {
  count: document.getElementById("knowledgeCount"),
  list: document.getElementById("knowledgeList"),
  enableAll: document.getElementById("enableAll"),
  disableAll: document.getElementById("disableAll"),
  search: document.getElementById("knowledgeSearch"),
};

/** The words searched: an event matches when its name, time or text contains all of them. */
const terms = () => state.query.toLowerCase().split(/\s+/).filter(Boolean);
const searchable = (event) =>
  `${displayName(event)} ${event.activity} ${timeLabel(event.offset_seconds)} ${event.summary} ${event.model_text}`.toLowerCase();

/** The events listed: every event of the case, or those matching the search. */
export function listedEvents() {
  const words = terms();
  const events = state.patient?.events || [];
  return words.length ? events.filter((event) => words.every((word) => searchable(event).includes(word))) : events;
}

/** The listed events known at the cutoff: the ones Select all and Clear act on. */
function listedKnown() {
  const known = new Set(availableEvents().map((event) => event.id));
  return listedEvents().filter((event) => known.has(event.id));
}

export function init() {
  els.list.addEventListener("change", (event) => {
    const item = event.target.closest("[data-knowledge]");
    if (item) toggleEvents([item.dataset.knowledge], event.target.checked);
  });
  // With a search, Select all and Clear act on the events listed
  els.enableAll.addEventListener("click", () => toggleEvents(listedKnown().map((event) => event.id), true));
  els.disableAll.addEventListener("click", () => toggleEvents(listedKnown().map((event) => event.id), false));
  els.search.addEventListener("input", () => {
    state.query = els.search.value;
    render();
  });
}

export function render() {
  const all = state.patient?.events || [];
  const known = new Set(availableEvents().map((event) => event.id));
  const listed = listedEvents();
  const searching = terms().length > 0;
  const later = all.length - known.size;
  els.count.textContent = all.length
    ? `${state.enabled.size} of ${all.length} selected${later ? `; ${later} after the cutoff` : ""}${searching ? `; ${listed.length} match` : ""}` : "";
  els.enableAll.title = searching ? "Select the events before the cutoff that match the search" : "Select every event before the cutoff";
  els.disableAll.title = searching ? "Clear the events before the cutoff that match the search" : "Clear every event";
  if (!all.length) {
    els.list.innerHTML = `<p class="empty-note">${escapeHtml(texts.events_empty)}</p>`;
    return;
  }
  if (!listed.length) {
    els.list.innerHTML = `<p class="empty-note">No event matches "${escapeHtml(state.query.trim())}".</p>`;
    return;
  }

  const explanation = state.explanation;
  const outside = new Set(explanation?.out_of_context_event_ids || []);
  const largest = explanation ? Math.max(...Object.values(explanation.events).map((entry) => Math.abs(entry.score))) : 1;
  const perDay = new Map();
  for (const event of listed) perDay.set(dayOf(event.offset_seconds), (perDay.get(dayOf(event.offset_seconds)) || 0) + 1);

  let currentDay = null;
  const rows = [];
  for (const event of listed) {
    const day = dayOf(event.offset_seconds);
    if (day !== currentDay) {
      rows.push(`<div class="knowledge-day">${texts.day} ${day}, ${plural(perDay.get(day), "event")}</div>`);
      currentDay = day;
    }
    const enabled = state.enabled.has(event.id);
    const isLater = !known.has(event.id);
    const score = explanation?.events[event.id]?.score;
    rows.push(`<label class="knowledge-item ${isLater ? "later" : enabled ? "" : "off"}" data-knowledge="${escapeHtml(event.id)}"${isLater ? ' title="After the cutoff: move the cutoff past it to give it to the models"' : ""}>
      <input type="checkbox" ${enabled && !isLater ? "checked" : ""} ${isLater ? "disabled" : ""}>
      <time>${clock(event.offset_seconds)}</time>
      <span>
        <strong><i style="--c: var(--lane-${eventTypeOf(event.activity).key})"></i>${escapeHtml(displayName(event))}${outside.has(event.id) ? '<span class="out-of-context" title="Outside the 512-token context the text models read">out of context</span>' : ""}${revealsOutcome(event) ? '<span class="out-of-context" title="This event states the outcome, so the models never get it">not given to the models</span>' : ""}</strong>
        <small title="${escapeHtml(event.model_text.trim())}">${escapeHtml(event.summary)}</small>
        ${score !== undefined ? `<span class="attr" title="Attribution: ${score.toFixed(3)}">${divergingBar(score, largest)}</span>` : ""}
      </span>
    </label>`);
  }
  const scroll = els.list.scrollTop;
  els.list.innerHTML = rows.join("");
  els.list.scrollTop = scroll;
}

export function reveal(eventId) {
  els.list.querySelector(`[data-knowledge="${CSS.escape(eventId)}"]`)?.scrollIntoView({ block: "nearest" });
}
