// "Download CSV": a snapshot of what the explorer shows for the open case, as one table.
// The section column says what a row is:
//   prediction   each model's prediction at the current cutoff (the Predictions panel)
//   trajectory   each point of the prediction lines under the timeline
//   explanation  each event's score in the last explanation (the Why? panel), if one was run
import { config, displayName, models, modelById, riskOutcome, texts } from "./config.js";
import { predictedClass } from "./metrics.js";
import { timeLabel } from "./format.js";
import { availableEvents, cutoffLabel, eventById, state } from "./state.js";
import { showToast } from "./views/toast.js";

const button = document.getElementById("exportCsv");
const probabilityColumns = config.outcomes.map((outcome) => `p_${outcome.code}`);
const COLUMNS = ["section", "case", "observed_outcome", "cutoff", "cutoff_time", "events_given", "model", "predicted",
  ...probabilityColumns, "matches_observed", "device", "inference_ms", "event", "event_time", "score"];

export function init() {
  button.addEventListener("click", download);
}

function rows() {
  const patient = state.patient;
  const observed = patient.ground_truth?.label ?? "";
  const base = { case: patient.id, observed_outcome: observed };
  const out = [];

  if (state.cutoff) {
    const cutoff = { cutoff: cutoffLabel(state.cutoff), cutoff_time: timeLabel(state.cutoff.seconds),
      events_given: `${state.enabled.size} of ${availableEvents().length}` };
    for (const model of models) {
      const result = state.results[model.id];
      if (!result) continue;
      const timing = result.performance || {};
      out.push({
        section: "prediction", ...base, ...cutoff, model: model.name, predicted: result.prediction.label,
        ...Object.fromEntries(config.outcomes.map((outcome) => [`p_${outcome.code}`, result.probabilities[outcome.code]])),
        matches_observed: result.ground_truth ? result.prediction.code === result.ground_truth.code : "",
        device: timing.device ?? "", inference_ms: timing.inference_ms ?? "",
      });
    }
  }

  for (const point of state.trajectory || []) {
    const where = point.kind === "day" ? `End of ${texts.day} ${point.day}` : displayName(eventById(point.stage_event_id));
    for (const model of models) {
      const risk = point.risk[model.id];
      if (risk === null || risk === undefined) continue;
      // As in the statistics: the most probable class
      const code = predictedClass(point, model.id);
      const predicted = config.outcomes.find((outcome) => outcome.code === code) || { code, label: code };
      out.push({
        section: "trajectory", ...base, cutoff: where, cutoff_time: timeLabel(point.offset_seconds), model: model.name,
        predicted: predicted.label, [`p_${riskOutcome.code}`]: risk,
        matches_observed: patient.ground_truth ? predicted.code === patient.ground_truth.code : "",
      });
    }
  }

  const explanation = state.explanation;
  if (explanation && state.cutoff) {
    for (const [eventId, entry] of Object.entries(explanation.events)) {
      const event = eventById(eventId);
      out.push({
        section: "explanation", ...base, cutoff: cutoffLabel(state.cutoff), cutoff_time: timeLabel(state.cutoff.seconds),
        model: modelById[explanation.model].name, predicted: explanation.prediction.label,
        event: displayName(event), event_time: timeLabel(event.offset_seconds), score: entry.score,
      });
    }
  }
  return out;
}

function cell(value) {
  const text = value === undefined || value === null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function download() {
  if (!state.patient) return showToast("Open a case first.");
  const lines = [COLUMNS.join(","), ...rows().map((row) => COLUMNS.map((column) => cell(row[column])).join(","))];
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const link = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob), download: `${state.patient.id.replace(/\W+/g, "-")}.csv`,
  });
  link.click();
  URL.revokeObjectURL(link.href);
}
