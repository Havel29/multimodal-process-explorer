// "Download CSV": what the open page shows, as one table.
// In the Case view, a snapshot of the open case; in Statistics, every computed case.
// The section column says what a row is:
//   prediction   each model's prediction at the current cutoff (the Predictions panel)
//   trajectory   each point of the prediction lines under the timeline
//   explanation  each event's score in the last explanation (the Why? panel), if one was run; positive
//                scores push toward the class shown on the page
//   importance   (Statistics) at each image cutoff, the change in each class probability (delta_<class>)
//                when every event of that type is removed (for the image models, the image is blanked);
//                score is the change for the class shown on the page
// A column only belongs to some sections, so the others leave it empty: device and inference time
// exist only for predictions, event and score only for explanations and importance.
import { api } from "./api.js";
import { capitalized, config, displayName, imageNoun, models, modelById, texts } from "./config.js";
import { classDelta, classProbability, importanceDelta, predictedClass } from "./metrics.js";
import { timeLabel } from "./format.js";
import { availableEvents, cutoffLabel, eventById, state } from "./state.js";
import { showToast } from "./views/toast.js";

const button = document.getElementById("exportCsv");
const probabilityColumns = config.outcomes.map((outcome) => `p_${outcome.code}`);
const deltaColumns = config.outcomes.map((outcome) => `delta_${outcome.code}`);
const COLUMNS = ["section", "case", "observed_outcome", "cutoff", "cutoff_time", "events_given", "model", "predicted",
  ...probabilityColumns, "matches_observed", "device", "inference_ms", "event", "event_time", "score", ...deltaColumns];

export function init() {
  button.addEventListener("click", () => (state.view === "statistics" ? downloadStatistics() : download()));
}

/** The rows of one case's prediction lines; where names the cutoff of a point. */
function trajectoryRows(points, base, truth, where) {
  const out = [];
  for (const point of points || []) {
    for (const model of models) {
      const risk = point.risk[model.id];
      if (risk === null || risk === undefined) continue;
      // As in the statistics: the most probable class
      const code = predictedClass(point, model.id);
      const predicted = config.outcomes.find((outcome) => outcome.code === code) || { code, label: code };
      out.push({
        section: "trajectory", ...base, cutoff: where(point), cutoff_time: timeLabel(point.offset_seconds),
        // Every event known at a point of the lines is given to the models
        events_given: point.event_count ? `${point.event_count} of ${point.event_count}` : "",
        model: model.name, predicted: predicted.label,
        ...Object.fromEntries(config.outcomes.map((outcome) => [`p_${outcome.code}`, classProbability(point, model.id, outcome.code)])),
        matches_observed: truth ? predicted.code === truth.code : "",
      });
    }
  }
  return out;
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

  out.push(...trajectoryRows(state.trajectory, base, patient.ground_truth, (point) =>
    (point.kind === "day" ? `End of ${texts.day} ${point.day}` : displayName(eventById(point.stage_event_id)))));

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

/** Every computed case of the Statistics page: its prediction lines and its event-type importance. */
async function statisticsRows() {
  const { patients } = await api.statistics();
  const imageCutoff = (offset) => `${capitalized(imageNoun)}, ${timeLabel(offset)}`;
  const out = [];
  for (const patient of patients) {
    const base = { case: patient.id, observed_outcome: patient.ground_truth?.label ?? "" };
    out.push(...trajectoryRows(patient.points, base, patient.ground_truth, (point) =>
      (point.kind === "day" ? `End of ${texts.day} ${point.day}` : imageCutoff(point.offset_seconds))));
    for (const profile of patient.importance || []) {
      for (const row of profile.rows) {
        for (const modelId of Object.keys(row.delta)) {
          out.push({
            section: "importance", ...base, cutoff: imageCutoff(profile.offset_seconds), cutoff_time: timeLabel(profile.offset_seconds),
            model: modelById[modelId]?.name ?? modelId, event: row.label, score: importanceDelta(row, modelId),
            ...Object.fromEntries(config.outcomes.map((outcome) => [`delta_${outcome.code}`, classDelta(row, modelId, outcome.code)])),
          });
        }
      }
    }
  }
  return out;
}

function save(tableRows, filename) {
  const lines = [COLUMNS.join(","), ...tableRows.map((row) => COLUMNS.map((column) => cell(row[column])).join(","))];
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: filename });
  link.click();
  URL.revokeObjectURL(link.href);
}

function download() {
  if (!state.patient) return showToast("Open a case first.");
  save(rows(), `${state.patient.id.replace(/\W+/g, "-")}.csv`);
}

async function downloadStatistics() {
  button.disabled = true;
  try {
    const tableRows = await statisticsRows();
    if (!tableRows.length) return showToast("No case is computed yet: use Compute missing cases first.");
    save(tableRows, "statistics.csv");
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
}
