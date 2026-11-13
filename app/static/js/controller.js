// User actions. Each one updates the state, then re-renders the views it affects.
import { api } from "./api.js";
import { DAY, eventTypeOf, models, setShownClass } from "./config.js";
import * as setup from "./setup.js";
import { availableEvents, endCutoff, imageEvent, selectionBody, state } from "./state.js";
import * as analysis from "./views/analysis.js";
import * as header from "./views/header.js";
import * as knowledge from "./views/knowledge.js";
import * as predictions from "./views/predictions.js";
import * as stage from "./views/stage.js";
import * as statistics from "./views/statistics.js";
import * as timeline from "./views/timeline.js";
import { showToast } from "./views/toast.js";

let patientSelectionToken = 0;

// One page per tab; only the active one is shown.
const PAGES = { patient: "patientView", statistics: "statisticsView", inputs: "setupWizard" };

function showPage(view) {
  state.view = view;
  for (const [key, id] of Object.entries(PAGES)) document.getElementById(id).classList.toggle("hidden", key !== view);
  // A reload stays on the same tab (the Inputs page reopens the Case view)
  if (view !== "inputs") history.replaceState(null, "", `#${view === "statistics" ? setup.STATISTICS_HASH : setup.EXPLORER_HASH}`);
  header.renderTabs();
}

/** Leaving the case view stops its running predictions and explanation; they restart on return. */
function leaveCaseView() {
  state.runToken += 1;
  state.analysisToken += 1;
}

export function renderAll() {
  timeline.render();
  stage.render();
  knowledge.render();
  predictions.renderButtons();
  analysis.renderButtons();
}

/**
 * Open a case with the cutoff at its last event and every event selected, and run the models.
 * Its prediction lines are computed afterwards, so they do not delay the first predictions.
 */
export async function selectPatient(patientId) {
  if (state.patientLoading || (state.view === "patient" && state.patient?.id === patientId)) return;
  const token = ++patientSelectionToken;
  state.patientLoading = true;
  header.renderTabs();
  try {
    const patient = await api.patient(patientId);
    if (token !== patientSelectionToken) return;
    Object.assign(state, {
      patient, cutoff: null, zoom: null, focused: null, enabled: new Set(),
      predicting: false, trajectory: null, trajectoryLoading: true,
      stayDays: Math.max(1, Math.ceil(patient.duration_seconds / DAY)),
    });
    state.runToken += 1;
    showPage("patient");
    timeline.closeBrush();
    analysis.clear();
    predictions.reset();
    header.render();
    setCutoff(endCutoff(), { run: false });
    void runPrediction().finally(() => {
      if (state.patient?.id === patient.id) void loadTrajectory(true);
    });
  } catch (error) {
    if (token === patientSelectionToken) showToast(error.message);
  } finally {
    if (token === patientSelectionToken) {
      state.patientLoading = false;
      header.renderTabs();
    }
  }
}

/** The Case view tab: back to the open case as it was left, or the first case. */
export async function showCaseView() {
  if (!state.patient) {
    if (state.patients.length) await selectPatient(state.patients[0].id);
    return;
  }
  showPage("patient");
  renderAll();  // the layout may have changed while the page was hidden
  // Predictions interrupted by leaving the page run again
  if (state.predicting) {
    state.predicting = false;
    void runPrediction();
  }
}

export async function showStatistics() {
  leaveCaseView();
  showPage("statistics");
  await statistics.show();
}

/** Another target class: the charts and statistics are drawn again, a running explanation is dropped
 * and the predictions of the open case run again. */
export function showClass(code) {
  if (!setShownClass(code)) return;
  analysis.clear();
  if (state.patient) {
    renderAll();
    header.renderInsights();
    if (state.view === "patient" && state.cutoff) runPrediction();
  }
  if (state.view === "statistics") statistics.rerender();
}

export function showInputs() {
  leaveCaseView();
  showPage("inputs");
  setup.open();
}

/** Move the cutoff; every event known at the new cutoff is given to the models. */
export function setCutoff(cutoff, { run = true } = {}) {
  state.cutoff = cutoff;
  state.enabled = new Set(availableEvents().map((event) => event.id));
  knowledgeChanged(run);
}

/** Set the cutoff and exactly which known events the models receive. */
export function setKnowledge(cutoff, enabledIds) {
  state.cutoff = cutoff;
  const known = new Set(availableEvents().map((event) => event.id));
  state.enabled = new Set([...enabledIds].filter((id) => known.has(id)));
  knowledgeChanged(true);
}

/** Tick or untick events; the predictions wait for "Run with selected events". */
export function toggleEvents(eventIds, enabled) {
  if (!state.cutoff) return;
  for (const id of eventIds) {
    if (enabled) state.enabled.add(id);
    else state.enabled.delete(id);
  }
  state.stale = true;
  analysis.clear();
  renderAll();
}

/** Turn every known event of one type off or on, then re-run. Without a cutoff, the whole stay is used. */
export function setEventType(typeKey, enabled) {
  if (!state.cutoff) setCutoff(endCutoff(), { run: false });
  for (const event of availableEvents()) {
    if (eventTypeOf(event.activity).key !== typeKey) continue;
    if (enabled) state.enabled.add(event.id);
    else state.enabled.delete(event.id);
  }
  knowledgeChanged(true);
}

function knowledgeChanged(run) {
  analysis.clear();
  renderAll();
  if (run) runPrediction();
}

/** Run every model in turn; results of superseded runs are discarded. */
export async function runPrediction() {
  if (!state.cutoff) return;
  const token = ++state.runToken;
  const hasImage = Boolean(imageEvent());
  const selection = selectionBody();
  state.predicting = true;
  state.stale = false;
  predictions.renderButtons();
  for (const model of models) predictions.showPending(model, hasImage);

  for (const model of models) {
    if (model.usesImage && !hasImage) continue;
    if (token !== state.runToken) return;
    predictions.showRunning(model);
    try {
      const result = await api.predict(selection, model.id);
      if (token === state.runToken) predictions.showResult(model, result);
    } catch (error) {
      if (token === state.runToken) predictions.showError(model, error.message);
    }
  }
  if (token === state.runToken) {
    state.predicting = false;
    predictions.renderButtons();
  }
}

export function focusEvent(eventId) {
  state.focused = eventId;
  timeline.render();
  timeline.describeEvent(eventId);
  knowledge.reveal(eventId);
}

export async function loadTrajectory(compute) {
  const patientId = state.patient.id;
  if (compute) {
    state.trajectoryLoading = true;
    timeline.render();
  }
  try {
    const { points } = await api.trajectory(patientId, compute);
    if (state.patient?.id === patientId) state.trajectory = points;
  } catch (error) {
    showToast(error.message);
  } finally {
    if (state.patient?.id === patientId) {
      state.trajectoryLoading = false;
      timeline.render();
      header.renderInsights();
    }
  }
}
