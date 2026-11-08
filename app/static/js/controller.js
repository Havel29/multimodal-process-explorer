// User actions. Each one updates the state, then re-renders the views it affects.
import { api } from "./api.js";
import { DAY, eventTypeOf, models } from "./config.js";
import { availableEvents, imageEvent, selectionBody, state } from "./state.js";
import * as analysis from "./views/analysis.js";
import * as header from "./views/header.js";
import * as knowledge from "./views/knowledge.js";
import * as predictions from "./views/predictions.js";
import * as stage from "./views/stage.js";
import * as statistics from "./views/statistics.js";
import * as timeline from "./views/timeline.js";
import { showToast } from "./views/toast.js";

let patientSelectionToken = 0;

export function renderAll() {
  timeline.render();
  stage.render();
  knowledge.render();
  predictions.renderButtons();
  analysis.renderButtons();
}

export async function selectPatient(patientId) {
  if (state.patientLoading || (state.view === "patient" && state.patient?.id === patientId)) return;
  const token = ++patientSelectionToken;
  state.patientLoading = true;
  header.renderTabs();
  try {
    const patient = await api.patient(patientId);
    if (token !== patientSelectionToken) return;
    Object.assign(state, {
      view: "patient", patient, cutoff: null, focused: null, enabled: new Set(),
      predicting: false, trajectory: null, trajectoryLoading: false,
      stayDays: Math.max(1, Math.ceil(patient.duration_seconds / DAY)),
    });
    state.runToken += 1;
    statistics.hide();
    timeline.closeBrush();
    analysis.clear();
    predictions.reset();
    header.render();
    renderAll();
    // Trajectories are expensive. Start a missing one in the background so the
    // case controls and Statistics remain usable while inference runs. The
    // patient-id check in loadTrajectory discards stale UI results after a
    // navigation, while the backend keeps the completed trajectory cached.
    void loadTrajectory(true);
  } catch (error) {
    if (token === patientSelectionToken) showToast(error.message);
  } finally {
    if (token === patientSelectionToken) {
      state.patientLoading = false;
      header.renderTabs();
    }
  }
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
  if (!state.cutoff) {
    const events = state.patient.events;
    setCutoff({ kind: "free", seconds: events[events.length - 1].offset_seconds }, { run: false });
  }
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

export async function showStatistics() {
  state.view = "statistics";
  state.runToken += 1;
  state.analysisToken += 1;
  header.renderTabs();
  await statistics.show();
}
