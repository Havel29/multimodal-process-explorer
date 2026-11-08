// Case navigation, title, observed outcome, and summary chips.
import { config, imageNoun, models, stageNoun, texts, fill, nouns } from "../config.js";
import { selectPatient, showStatistics } from "../controller.js";
import { escapeHtml, plural, timeLabel } from "../format.js";
import { agreement, flips, series, settlesFrom } from "../metrics.js";
import { state } from "../state.js";

const els = {
  patientSelect: document.getElementById("patientSelect"),
  previousPatient: document.getElementById("previousPatient"),
  nextPatient: document.getElementById("nextPatient"),
  statisticsTab: document.getElementById("statisticsTab"),
  title: document.getElementById("patientTitle"),
  meta: document.getElementById("patientMeta"),
  truth: document.getElementById("truthLabel"),
  insights: document.getElementById("insights"),
};

export function init() {
  els.patientSelect.addEventListener("change", () => selectPatient(els.patientSelect.value));
  els.previousPatient.addEventListener("click", () => movePatient(-1));
  els.nextPatient.addEventListener("click", () => movePatient(1));
  if (config.hasLabels) els.statisticsTab.addEventListener("click", showStatistics);
}

export function renderTabs() {
  const selected = state.patient?.id || "";
  els.patientSelect.innerHTML = state.patients.map((patient) =>
    `<option value="${escapeHtml(patient.id)}" title="${plural(patient.event_count, "event")}, ${plural(patient.prediction_stage_count, imageNoun)}"
      ${patient.id === selected ? "selected" : ""}>${escapeHtml(patient.id)}</option>`).join("");
  const index = state.patients.findIndex((patient) => patient.id === selected);
  els.patientSelect.disabled = state.patientLoading;
  els.previousPatient.disabled = state.patientLoading || index <= 0;
  els.nextPatient.disabled = state.patientLoading || index < 0 || index >= state.patients.length - 1;
  els.statisticsTab.disabled = state.patientLoading;
  els.statisticsTab.classList.toggle("active", state.view === "statistics");
}

function movePatient(offset) {
  if (state.patientLoading) return;
  const index = state.patients.findIndex((patient) => patient.id === state.patient?.id);
  const patient = state.patients[index + offset];
  if (patient) selectPatient(patient.id);
}

export function render() {
  const patient = state.patient;
  const images = patient.events.filter((event) => event.prediction_stage).length;
  els.title.textContent = patient.id;
  els.meta.textContent = fill("case_meta", {
    event_count: plural(patient.events.length, "event"), day_count: plural(state.stayDays, nouns.day), image_count: plural(images, imageNoun),
  });
  if (config.hasLabels) {
    els.truth.textContent = patient.ground_truth.label;
    els.truth.style.setProperty("--tone", `var(--outcome-${patient.ground_truth.code})`);
  }
  renderTabs();
  renderInsights();
}

/** Earliness, stability and agreement of the models for this case. */
export function renderInsights() {
  const points = state.trajectory;
  if (!config.hasLabels || !points || !state.patient) {
    els.insights.innerHTML = "";
    return;
  }
  const truth = state.patient.ground_truth.code;
  const chips = models.filter((model) => series(points, model.id).length).map((model) => {
    const settled = settlesFrom(points, model.id, truth);
    const count = flips(points, model.id);
    const where = settled
      ? `right from ${settled.kind === "day" ? `the end of ${texts.day} ${settled.day}` : `the ${stageNoun} on ${timeLabel(settled.offset_seconds)}`}`
      : "never settles on the observed outcome";
    return `<span class="chip ${settled ? "" : "bad"}" title="Earliness: the first cutoff after which every later prediction matches the observed outcome. Flips: changes of predicted class along the pathway.">
      <i class="swatch" style="background: var(--model-${model.id})"></i><strong>${escapeHtml(model.name)}</strong> ${where}; ${plural(count, "flip")}</span>`;
  });
  const agree = models.length > 1 && agreement(points);  // with one model there is nothing to agree on
  if (agree) chips.push(`<span class="chip" title="Cutoffs where every model can run and all predict the same class">All models agree at ${agree.agree} of ${agree.total} cutoffs</span>`);
  els.insights.innerHTML = chips.join("");
}
