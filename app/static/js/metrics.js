// Metrics over the prediction lines. A prediction is the most probable class (any number of classes);
// AUC and Brier score compare the class shown on the page against all the others.
import { config, configuredOther, configuredRisk, models, riskOutcome } from "./config.js";

/** The class a model predicts at a point (older points without "predicted": two classes, p(risk) >= 0.5). */
export function predictedClass(point, model) {
  const stored = point.predicted?.[model];
  if (stored) return stored;
  const risk = point.risk[model];
  return risk === null || risk === undefined ? null : risk >= 0.5 ? configuredRisk : configuredOther.code;
}

/**
 * The probability a model gives one class at a point: null where the model cannot run, undefined
 * where it is not stored (lines computed before every class was kept, with more than two classes).
 */
export function classProbability(point, model, code) {
  const risk = point.risk[model];
  if (risk === null || risk === undefined) return null;
  const stored = point.probabilities?.[model]?.[code];
  if (stored !== undefined) return stored;
  if (code === configuredRisk) return risk;
  return config.outcomes.length === 2 ? 1 - risk : undefined;
}

/** The probability of the class shown on the page. */
export const probability = (point, model) => classProbability(point, model, riskOutcome.code);

/** Whether some line lacks the shown class, which then needs computing again. */
export const missingShownClass = (points) => (points || []).some((point) =>
  models.some((model) => probability(point, model.id) === undefined));

/** The change in one class's probability in a row of the event-type importance (as classProbability). */
export function classDelta(row, model, code) {
  const stored = row.deltas?.[code]?.[model];
  if (stored !== undefined) return stored;
  const delta = row.delta?.[model];
  if (delta === undefined) return undefined;
  if (code === configuredRisk) return delta;
  return config.outcomes.length === 2 ? -delta : undefined;
}

/** The change in the shown class's probability. */
export const importanceDelta = (row, model) => classDelta(row, model, riskOutcome.code);

export const isCorrect = (point, model, truth) => predictedClass(point, model) === truth;

export function series(points, model) {
  return (points || [])
    .filter((point) => point.risk[model] !== null && point.risk[model] !== undefined)
    .sort((a, b) => a.offset_seconds - b.offset_seconds);
}

/** Earliness: the first point from which every later prediction is correct (null if none). */
export function settlesFrom(points, model, truth) {
  const values = series(points, model);
  let index = values.length;
  while (index > 0 && isCorrect(values[index - 1], model, truth)) index -= 1;
  return index < values.length ? values[index] : null;
}

/** How often the predicted class changes along the pathway. */
export function flips(points, model) {
  const values = series(points, model).map((point) => predictedClass(point, model));
  return values.slice(1).filter((value, index) => value !== values[index]).length;
}

/** Points where every model can run, and how many of them all models agree on. */
export function agreement(points) {
  const shared = (points || []).filter((point) => models.every((model) => point.risk[model.id] !== null));
  const agree = shared.filter((point) => new Set(models.map((model) => predictedClass(point, model.id))).size === 1);
  return shared.length ? { agree: agree.length, total: shared.length } : null;
}

const scored = (pairs) => pairs.filter((pair) => pair.p !== undefined && pair.p !== null);

/** ROC AUC by pairwise comparison (Mann-Whitney); null without both classes. */
export function auc(pairs) {
  const positives = scored(pairs).filter((pair) => pair.y === 1);
  const negatives = scored(pairs).filter((pair) => pair.y === 0);
  if (!positives.length || !negatives.length) return null;
  let wins = 0;
  for (const positive of positives) {
    for (const negative of negatives) wins += positive.p > negative.p ? 1 : positive.p === negative.p ? 0.5 : 0;
  }
  return wins / (positives.length * negatives.length);
}

export const accuracy = (pairs) => (pairs.length ? pairs.filter((pair) => pair.predicted === pair.truth).length / pairs.length : null);

/** Macro F1 over the classes that are observed or predicted. */
export function macroF1(pairs) {
  if (!pairs.length) return null;
  const classes = [...new Set(pairs.flatMap((pair) => [pair.truth, pair.predicted]))];
  const scores = classes.map((code) => {
    const truePositives = pairs.filter((pair) => pair.predicted === code && pair.truth === code).length;
    const errors = pairs.filter((pair) => (pair.predicted === code) !== (pair.truth === code)).length;
    return truePositives || errors ? (2 * truePositives) / (2 * truePositives + errors) : 0;
  });
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

/** Brier score: mean squared gap between the predicted probability and the outcome (0 or 1). */
export function brier(pairs) {
  const values = scored(pairs);
  return values.length ? values.reduce((sum, pair) => sum + (pair.p - pair.y) ** 2, 0) / values.length : null;
}

/** Probability of the shown class, whether it is the observed one (1/0), predicted and observed class
 *  over patients, for one model, at the points chosen by `pick`. */
export function pairs(patients, model, pick = (points) => points) {
  return patients.flatMap((patient) =>
    series(pick(patient.points), model).map((point) => ({
      p: probability(point, model), y: patient.ground_truth.code === riskOutcome.code ? 1 : 0,
      predicted: predictedClass(point, model), truth: patient.ground_truth.code,
    })),
  );
}

export const atDay = (day) => (points) => points.filter((point) => point.kind === "day" && point.day === day);

/** The image cutoffs where every model runs: one prediction per image, the same points for all models. */
export const atSharedImages = (points) => points.filter((point) => point.kind === "stage"
  && models.every((model) => point.risk[model.id] !== null && point.risk[model.id] !== undefined));
