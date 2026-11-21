// One result row per model, filled in as the models run.
import { config, imageNoun } from "../config.js";
import { runPrediction } from "../controller.js";
import { escapeHtml, percent } from "../format.js";
import { state } from "../state.js";

const button = document.getElementById("predictButton");
const row = (model) => document.querySelector(`.model-row[data-model="${model.id}"]`);

export function init() {
  button.addEventListener("click", runPrediction);
}

function setRow(model, html, unavailable = false) {
  delete state.results[model.id];  // showResult stores it again
  const element = row(model);
  element.classList.toggle("unavailable", unavailable);
  element.querySelector(".model-outcome").innerHTML = html;
}

export function reset() {
  for (const model of config.models) setRow(model, "");
  state.stale = false;
  renderButtons();
}

export function showPending(model, hasImage) {
  if (model.usesImage && !hasImage) setRow(model, `Needs a ${imageNoun} at this cutoff`, true);
  else setRow(model, "Waiting");
}

export const showRunning = (model) => setRow(model, "Running");
export const showError = (model, message) => setRow(model, escapeHtml(message), true);

export function showResult(model, result) {
  const predicted = result.prediction.code;
  const split = config.outcomes.map((outcome) =>
    `<span style="width: ${result.probabilities[outcome.code] * 100}%; background: var(--outcome-${outcome.code})"></span>`).join("");
  const breakdown = config.outcomes.map((outcome) => `${outcome.label} ${percent(result.probabilities[outcome.code])}`).join(", ");
  const comparison = result.ground_truth
    ? `<div class="match">${predicted === result.ground_truth.code ? "Matches" : "Differs from"} the observed outcome</div>`
    : "";
  const timing = result.performance;
  const speed = timing
    ? `<div class="performance" title="Total request ${timing.total_ms.toFixed(1)} ms${timing.cold_start ? `; model load ${timing.load_ms.toFixed(1)} ms` : ""}">${escapeHtml(timing.device.toUpperCase())} inference ${timing.inference_ms.toFixed(1)} ms${timing.cold_start ? " (cold start)" : ""}</div>`
    : "";
  setRow(model, `
    <div class="verdict"><strong style="color: var(--outcome-${predicted})">${escapeHtml(result.prediction.label)}</strong><span>${percent(result.probabilities[predicted])}</span></div>
    <div class="split" title="${escapeHtml(breakdown)}">${split}</div>
    ${comparison}${speed}`);
  state.results[model.id] = result;
}

export function renderButtons() {
  button.disabled = state.predicting || !state.cutoff;
  button.textContent = state.predicting ? "Running" : state.stale ? "Run with selected events" : "Run again";
}
