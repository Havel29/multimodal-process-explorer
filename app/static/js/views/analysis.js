// "Why?": the explanation of a model, one button per model with an explanation in the configuration.
import { api } from "../api.js";
import { config, displayName, imageNoun, modelById, riskOutcome, fill, texts } from "../config.js";
import { focusEvent } from "../controller.js";
import { escapeHtml, percent, plural, timeLabel } from "../format.js";
import { eventById, imageEvent, selectionBody, state } from "../state.js";
import { divergingBar } from "./bars.js";
import * as knowledge from "./knowledge.js";
import * as stage from "./stage.js";
import * as timeline from "./timeline.js";

const TOP_EVENTS = 8;
const explanations = Object.fromEntries(config.explanations.map((item) => [item.model, item]));

const els = {
  buttons: [...document.querySelectorAll("[data-explain]")],
  scope: document.getElementById("explainScope"),
  result: document.getElementById("analysisResult"),
};

const methods = [...new Set(config.explanations.map((item) => item.method))].join(" or ");
const INTRO = `<p class="empty-note">${escapeHtml(fill("explain_intro", {
  model: config.explanations.map((item) => modelById[item.model].name).join(", "), method: methods }))}</p>`;

export function init() {
  for (const button of els.buttons) button.addEventListener("click", () => run(button.dataset.explain));
  els.result.addEventListener("click", (event) => {
    const row = event.target.closest("[data-focus]");
    if (row) focusEvent(row.dataset.focus);
  });
  clear();
}

export function clear() {
  state.analysisToken += 1;
  state.analysisBusy = false;
  state.explanation = null;
  els.result.innerHTML = INTRO;
  renderButtons();
}

export function renderButtons() {
  const ready = Boolean(state.cutoff) && state.enabled.size > 0 && !state.analysisBusy;
  for (const button of els.buttons) {
    const model = modelById[button.dataset.explain];
    const needsImage = model.usesImage && !imageEvent();
    button.disabled = !ready || needsImage;
    button.title = needsImage ? `Needs a ${imageNoun} at this cutoff` : "";
    button.textContent = state.analysisBusy === model.id ? "Explaining" : `Explain ${model.name}`;
  }
  els.scope.disabled = Boolean(state.analysisBusy);
}

async function run(modelId) {
  const token = ++state.analysisToken;
  const model = modelById[modelId];
  // The whole history only means something for a model that reads text
  const scope = model.usesText ? els.scope.value : "context";
  const reads = model.usesText && model.usesImage ? "text and image" : model.usesText ? "the text" : `the ${imageNoun}`;
  state.analysisBusy = modelId;
  renderButtons();
  els.result.innerHTML = `<p class="empty-note">Running ${escapeHtml(explanations[modelId].method)} on ${escapeHtml(model.name)} over ${reads}${scope === "history"
    ? ", then over every earlier part of the history. This takes about half a minute per part on CPU."
    : ". This takes about half a minute on CPU."}</p>`;
  try {
    // Positive scores push toward the class shown on the page
    const result = await api.explain(selectionBody(), scope, modelId, riskOutcome.code);
    if (token !== state.analysisToken) return;
    state.explanation = result;
    renderExplanation(result);
    knowledge.render();
    timeline.render();
    stage.drawHeatmap();
    els.result.scrollIntoView({ block: "nearest" });
  } catch (error) {
    if (token === state.analysisToken) els.result.innerHTML = `<p class="empty-note">${escapeHtml(error.message)}</p>`;
  } finally {
    if (token === state.analysisToken) {
      state.analysisBusy = false;
      renderButtons();
    }
  }
}

// A function: the class shown can change
const direction = () => `Bars point <span class="toward-risk-text">right, toward ${escapeHtml(riskOutcome.short)}</span>, or
  <span class="away-from-risk-text">left, away from it</span>.`;

function scopeNote(result) {
  const outside = result.out_of_context_event_ids.length;
  if (result.scope === "history") {
    return `<p class="note">Whole history: the text was explained in ${plural(result.windows, "part")} of up to 512 tokens,
      each for the class the model predicted, and the scores were mapped back to events. ${outside
        ? `${plural(outside, "event")} marked "out of context" also get scores here, but the model itself never reads them.`
        : "Every event fits in the model's context."} Scores are normalised within each part.</p>`;
  }
  return outside ? `<p class="note">${plural(outside, "event")} fell outside the 512-token context the model reads (it keeps the
    most recent text), so they have no score. They are marked "out of context". Choose "Whole history" to score them anyway.</p>` : "";
}

/** The context the model read is short: show its text, each word shaded by its attribution. */
function readText(result) {
  const read = state.patient.events.filter((event) => result.events[event.id]);
  const scores = read.flatMap((event) => result.events[event.id].segments.map(([, score]) => Math.abs(score || 0)))
    .filter(Boolean).sort((a, b) => a - b);
  // A few words often score far above the rest: scale by the 95th percentile so the others stay visible
  const largest = scores[Math.floor(scores.length * 0.95)] || 1;
  return read.map((event) => {
    // The context keeps the most recent text, so the earliest event may be read only in part: show that part
    const segments = result.events[event.id].segments;
    const first = segments.findIndex(([, score]) => score !== null);
    const cut = first > 0 && segments.slice(0, first).map(([piece]) => piece).join("").length > 40;
    const text = (cut ? ["... "] : []).concat(segments.slice(cut ? first : 0).map(([piece, score]) => {
      if (score === null || score === 0) return escapeHtml(piece);
      const share = Math.round(Math.min(1, Math.sqrt(Math.abs(score) / largest)) * 85);
      return `<mark class="${score > 0 ? "toward-risk" : "away-from-risk"}" style="--share: ${share}%" title="${score.toFixed(3)}">${escapeHtml(piece)}</mark>`;
    })).join("");
    return `<button class="read-event" data-focus="${escapeHtml(event.id)}" type="button">
      <strong>${escapeHtml(displayName(event))}</strong><small>${escapeHtml(timeLabel(event.offset_seconds))}</small>
      <p>${text}</p></button>`;
  }).join("");
}

function renderExplanation(result) {
  if (!modelById[result.model].usesText) {
    // An image-only model: its attributions are the overlay on the image
    els.result.innerHTML = `
      <p>${escapeHtml(modelById[result.model].name)} predicts <strong>${escapeHtml(result.prediction.label)}</strong> (${percent(result.probabilities[result.prediction.code])}).
        It reads only the ${escapeHtml(imageNoun)}, so its attributions are the overlay on it: regions in red push the
        prediction toward ${escapeHtml(riskOutcome.short)}, in blue away from it.</p>
      ${methodNote(result)}`;
    return;
  }
  if (result.scope === "context") {
    els.result.innerHTML = `
      <p>${escapeHtml(modelById[result.model].name)} predicts <strong>${escapeHtml(result.prediction.label)}</strong> (${percent(result.probabilities[result.prediction.code])}).
        ${escapeHtml(fill("context_legend", { risk: riskOutcome.short }))}
        <span class="key"><mark class="toward-risk">toward</mark> <mark class="away-from-risk">away</mark></span></p>
      <h3>${escapeHtml(texts.context_title)}</h3>
      <div class="read-list">${readText(result) || '<p class="empty-note">No event text inside the model context.</p>'}</div>
      ${scopeNote(result)}
      ${methodNote(result)}`;
    return;
  }
  const outside = new Set(result.out_of_context_event_ids);
  const ranked = Object.entries(result.events)
    .sort(([, a], [, b]) => Math.abs(b.score) - Math.abs(a.score))
    .slice(0, TOP_EVENTS);
  const largest = Math.abs(ranked[0]?.[1].score || 1);
  const rows = ranked.map(([eventId, entry]) => {
    const event = eventById(eventId);
    const words = entry.tokens.map((token) => token.text).join(", ");
    return `<button class="attr-row" data-focus="${escapeHtml(eventId)}" type="button" title="${escapeHtml(event.model_text.trim())}">
      <span>${escapeHtml(displayName(event))}${outside.has(eventId) ? ' <em class="out-of-context">out of context</em>' : ""}
      <small>${escapeHtml(timeLabel(event.offset_seconds))}${words ? `; ${escapeHtml(words)}` : ""}</small></span>
      ${divergingBar(entry.score, largest)}</button>`;
  }).join("");
  els.result.innerHTML = `
    <p>${escapeHtml(modelById[result.model].name)} predicts <strong>${escapeHtml(result.prediction.label)}</strong> (${percent(result.probabilities[result.prediction.code])}). ${direction()}</p>
    <h3>${escapeHtml(texts.history_title)}</h3>
    <div class="attr-list">${rows || '<p class="empty-note">No event text inside the model context.</p>'}</div>
    ${scopeNote(result)}
    ${methodNote(result)}`;
}

function methodNote(result) {
  const steps = result.method === "Layer Integrated Gradients" ? ` with ${result.steps} steps` : "";
  const reference = explanations[result.model]?.reference ? `, as in ${escapeHtml(explanations[result.model].reference)}` : "";
  return `<p class="note">${result.patches.length ? `The ${imageNoun} overlay shows the image attributions, scaled on their own: compare colours within a modality, not across.` : ""}Method: ${escapeHtml(result.method)}${steps}${reference}.</p>`;
}
