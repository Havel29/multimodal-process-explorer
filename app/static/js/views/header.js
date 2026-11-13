// Tabs, case navigation, title, observed outcome, and summary chips.
import { config, imageNoun, models, riskOutcome, stageNoun, texts, fill, nouns } from "../config.js";
import { selectPatient, showCaseView, showClass, showInputs, showStatistics } from "../controller.js";
import { escapeHtml, plural, timeLabel } from "../format.js";
import { agreement, flips, series, settlesFrom } from "../metrics.js";
import { state } from "../state.js";

const els = {
  caseTab: document.getElementById("caseTab"),
  statisticsTab: document.getElementById("statisticsTab"),
  inputsTab: document.getElementById("inputsTab"),
  picker: document.querySelector(".patient-picker"),
  patientSelect: document.getElementById("patientSelect"),
  previousPatient: document.getElementById("previousPatient"),
  nextPatient: document.getElementById("nextPatient"),
  exportCsv: document.getElementById("exportCsv"),
  device: document.getElementById("deviceSelect"),
  classControl: document.getElementById("classControl"),
  classSelect: document.getElementById("classSelect"),
  help: document.getElementById("helpButton"),
  title: document.getElementById("patientTitle"),
  meta: document.getElementById("patientMeta"),
  truth: document.getElementById("truthLabel"),
  insights: document.getElementById("insights"),
};

const CSV_TITLES = {
  patient: "The predictions, prediction lines and explanation of this case, as a CSV file",
  statistics: "The prediction lines and event-type importance of every computed case, as a CSV file",
  inputs: "Open the Case view or Statistics to download their data",
};

export function init() {
  els.caseTab.addEventListener("click", showCaseView);
  if (config.hasLabels) els.statisticsTab.addEventListener("click", showStatistics);
  els.inputsTab.addEventListener("click", showInputs);
  els.patientSelect.addEventListener("change", () => selectPatient(els.patientSelect.value));
  els.previousPatient.addEventListener("click", () => movePatient(-1));
  els.nextPatient.addEventListener("click", () => movePatient(1));
  els.classSelect.addEventListener("change", () => showClass(els.classSelect.value));
  fillClasses();
}

/** The class selector: every outcome class, starting at the one shown (the risk class by default). */
function fillClasses() {
  els.classSelect.innerHTML = config.outcomes.map((outcome) =>
    `<option value="${escapeHtml(outcome.code)}">${escapeHtml(outcome.label)}</option>`).join("");
  els.classSelect.value = riskOutcome.code;
  els.classControl.classList.toggle("hidden", config.outcomes.length < 2);
}

/**
 * Before the explorer is opened only the Inputs tab is live: the other tabs open the explorer
 * on the data already loaded, or wait for a setup.
 */
export function initSetupPage(configured, openExplorer) {
  state.view = "inputs";
  for (const [tab, page] of [[els.caseTab, "explorer"], [els.statisticsTab, "statistics"]]) {
    tab.disabled = !configured;
    tab.title = configured ? "Open the explorer on the data already loaded" : "Complete the inputs first";
    tab.addEventListener("click", () => openExplorer(page));
  }
  els.inputsTab.classList.add("active");
  els.inputsTab.setAttribute("aria-current", "page");
  fillClasses();
  for (const control of [els.exportCsv, els.device, els.help, els.classSelect]) {
    control.disabled = true;
    control.title = "Available once the explorer is open";
  }
}

export function renderTabs() {
  for (const [tab, view] of [[els.caseTab, "patient"], [els.statisticsTab, "statistics"], [els.inputsTab, "inputs"]]) {
    tab.classList.toggle("active", state.view === view);
    if (state.view === view) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
  // The case picker only belongs to the Case view
  els.picker.classList.toggle("hidden", state.view !== "patient");
  els.exportCsv.disabled = state.view === "inputs";
  els.exportCsv.title = CSV_TITLES[state.view];

  const selected = state.patient?.id || "";
  els.patientSelect.innerHTML = state.patients.map((patient) =>
    `<option value="${escapeHtml(patient.id)}" title="${plural(patient.event_count, "event")}, ${plural(patient.prediction_stage_count, imageNoun)}"
      ${patient.id === selected ? "selected" : ""}>${escapeHtml(patient.id)}</option>`).join("");
  const index = state.patients.findIndex((patient) => patient.id === selected);
  els.patientSelect.disabled = state.patientLoading;
  els.previousPatient.disabled = state.patientLoading || index <= 0;
  els.nextPatient.disabled = state.patientLoading || index < 0 || index >= state.patients.length - 1;
  els.statisticsTab.disabled = state.patientLoading;
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
