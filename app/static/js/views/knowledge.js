// The list of events known at the cutoff, with a checkbox per event.
import { displayName, eventTypeOf, revealsOutcome, texts } from "../config.js";
import { toggleEvents } from "../controller.js";
import { clock, dayOf, escapeHtml, plural } from "../format.js";
import { availableEvents, state } from "../state.js";
import { divergingBar } from "./bars.js";

const els = {
  count: document.getElementById("knowledgeCount"),
  list: document.getElementById("knowledgeList"),
  enableAll: document.getElementById("enableAll"),
  disableAll: document.getElementById("disableAll"),
};

export function init() {
  els.list.addEventListener("change", (event) => {
    const item = event.target.closest("[data-knowledge]");
    if (item) toggleEvents([item.dataset.knowledge], event.target.checked);
  });
  els.enableAll.addEventListener("click", () => toggleEvents(availableEvents().map((event) => event.id), true));
  els.disableAll.addEventListener("click", () => toggleEvents(availableEvents().map((event) => event.id), false));
}

export function render() {
  const available = availableEvents();
  els.count.textContent = available.length ? `${state.enabled.size} of ${available.length} selected` : "";
  if (!available.length) {
    els.list.innerHTML = `<p class="empty-note">${escapeHtml(texts.events_empty)}</p>`;
    return;
  }

  const explanation = state.explanation;
  const outside = new Set(explanation?.out_of_context_event_ids || []);
  const largest = explanation ? Math.max(...Object.values(explanation.events).map((entry) => Math.abs(entry.score))) : 1;
  const perDay = new Map();
  for (const event of available) perDay.set(dayOf(event.offset_seconds), (perDay.get(dayOf(event.offset_seconds)) || 0) + 1);

  let currentDay = null;
  const rows = [];
  for (const event of available) {
    const day = dayOf(event.offset_seconds);
    if (day !== currentDay) {
      rows.push(`<div class="knowledge-day">${texts.day} ${day}, ${plural(perDay.get(day), "event")}</div>`);
      currentDay = day;
    }
    const enabled = state.enabled.has(event.id);
    const score = explanation?.events[event.id]?.score;
    rows.push(`<label class="knowledge-item ${enabled ? "" : "off"}" data-knowledge="${escapeHtml(event.id)}">
      <input type="checkbox" ${enabled ? "checked" : ""}>
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
