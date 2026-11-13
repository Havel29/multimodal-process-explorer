// Application state and the questions every view asks about it.
import { DAY, displayName, shortName, texts } from "./config.js";
import { plural, timeLabel } from "./format.js";

export const state = {
  view: "patient",  // "patient" (Case view), "statistics" or "inputs"
  patients: [],
  patient: null,
  patientLoading: false,
  stayDays: 1,
  // { kind: "day" | "stage" | "free", seconds, day?, stageId? }
  cutoff: null,
  zoom: null,  // the time window shown on the timeline, { start, end } in seconds; null: the whole case
  query: "",  // the search over "Events given to the models"
  enabled: new Set(),
  focused: null,
  predicting: false,
  stale: false,
  runToken: 0,
  trajectory: null,
  results: {},  // the latest prediction of each model at the current cutoff (for the CSV export)
  trajectoryLoading: false,
  explanation: null,
  analysisBusy: false,
  analysisToken: 0,
};

export const eventById = (eventId) => state.patient.events.find((event) => event.id === eventId);

export function availableEvents(cutoff = state.cutoff) {
  if (!state.patient || !cutoff) return [];
  const events = state.patient.events;
  if (cutoff.kind === "stage") return events.slice(0, events.findIndex((event) => event.id === cutoff.stageId) + 1);
  return events.filter((event) => event.offset_seconds <= cutoff.seconds);
}

/** The image models use the latest enabled image known at the cutoff. */
export function imageEvent(cutoff = state.cutoff) {
  return [...availableEvents(cutoff)].reverse()
    .find((event) => event.prediction_stage && state.enabled.has(event.id)) || null;
}

export function cutoffLabel(cutoff) {
  const count = plural(availableEvents(cutoff).length, "event");
  if (cutoff.kind === "day") return `End of ${texts.day} ${cutoff.day} (${count})`;
  if (cutoff.kind === "stage") {
    const stage = eventById(cutoff.stageId);
    return `${shortName(stage) || displayName(stage)}, ${timeLabel(stage.offset_seconds)} (${count})`;
  }
  return `${timeLabel(cutoff.seconds)} (${count})`;
}

/** The request body describing the current cutoff and event selection. */
export function selectionBody() {
  const body = { patient_id: state.patient.id, enabled_event_ids: [...state.enabled] };
  if (state.cutoff.kind === "day") body.cutoff_day = state.cutoff.day;
  else if (state.cutoff.kind === "stage") body.stage_event_id = state.cutoff.stageId;
  else body.cutoff_seconds = Math.round(state.cutoff.seconds);
  return body;
}

/** The cutoff at the last event of the case: the models see the whole trace. */
export function endCutoff() {
  const events = state.patient.events;
  return { kind: "free", seconds: events[events.length - 1].offset_seconds };
}

/** True when the cutoff is at the last event and every event is selected. */
export function atEnd() {
  if (!state.cutoff || !state.patient) return false;
  return state.cutoff.seconds >= endCutoff().seconds && state.enabled.size === availableEvents().length;
}

export const dayCutoff = (day) => ({ kind: "day", day, seconds: day * DAY });
export const stageCutoff = (event) => ({ kind: "stage", stageId: event.id, seconds: event.offset_seconds });
