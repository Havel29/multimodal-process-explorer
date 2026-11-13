// The dataset and model configuration, embedded in the page by the server.

export const config = JSON.parse(document.getElementById("demo-config").textContent);
export const DAY = 86400;

/** How the interface names images ("radiograph") and the events carrying them ("X-ray"). */
export const imageNoun = config.imageNoun;
export const stageNoun = config.stageNoun;
export const capitalized = (text) => text.charAt(0).toUpperCase() + text.slice(1);

const activities = new Map();
for (const type of config.eventTypes) {
  for (const activity of type.activities) activities.set(activity.name, { ...activity, type });
}
const otherType = config.eventTypes.find((type) => type.key === "other");

export const models = config.models;
export const modelById = Object.fromEntries(models.map((model) => [model.id, model]));
// Before the setup a configuration may have no outcome classes yet: they come from the event log
const PLACEHOLDER_RISK = { code: "", label: "risk outcome", short: "risk", color: "#e34948" };
const PLACEHOLDER_OTHER = { code: "", label: "other outcome", short: "other", color: "#2a78d6" };
const outcomeOf = (code) => config.outcomes.find((outcome) => outcome.code === code);
const otherThan = (code) => config.outcomes.find((outcome) => outcome.code !== code);

/** The risk class of the configuration (by default the rarest): what the server computed "risk" for. */
export const configuredRisk = config.riskClass;
/** The other class of the configuration, for prediction lines saved before the predicted class was stored. */
export const configuredOther = otherThan(configuredRisk) || PLACEHOLDER_OTHER;

// The class whose probability the charts, statistics and explanations show. It starts at the
// configured risk class and can be changed from the page (setShownClass); the bindings are live.
export let riskOutcome = outcomeOf(configuredRisk) || PLACEHOLDER_RISK;
export let otherOutcome = otherThan(configuredRisk) || PLACEHOLDER_OTHER;
export let riskLabel = `P(${riskOutcome.short})`;

export function eventTypeOf(activity) {
  return activities.get(activity)?.type || otherType;
}

export function displayName(event) {
  return activities.get(event.activity)?.display || event.activity;
}

/** Events that state the outcome (e.g. the discharge): on the timeline, but never given to the models. */
export function revealsOutcome(event) {
  return Boolean(activities.get(event.activity)?.revealsOutcome);
}

export function shortName(event) {
  return activities.get(event.activity)?.short || "";
}

export function rgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export let riskRgb = rgb(riskOutcome.color);
export let safeRgb = rgb(otherOutcome.color);

const SHOWN_CLASS_KEY = "explorer.shownClass";

/** Show another class: its probability on the charts and statistics, and explanations toward it. */
export function setShownClass(code) {
  const outcome = outcomeOf(code);
  if (!outcome) return false;
  riskOutcome = outcome;
  otherOutcome = otherThan(code) || PLACEHOLDER_OTHER;
  riskLabel = `P(${riskOutcome.short})`;
  riskRgb = rgb(riskOutcome.color);
  safeRgb = rgb(otherOutcome.color);
  document.documentElement.style.setProperty("--risk", `var(--outcome-${code})`);
  try {
    localStorage.setItem(SHOWN_CLASS_KEY, code);
  } catch (_) {
    // Private windows may refuse storage; the choice then lasts until a reload
  }
  return true;
}

/** The class chosen last time in this browser, if the data still has it. */
export function restoreShownClass() {
  try {
    const saved = localStorage.getItem(SHOWN_CLASS_KEY);
    if (saved && saved !== riskOutcome.code) setShownClass(saved);
  } catch (_) {
    // Nothing saved
  }
}

/** The texts of the interface ([texts] in the configuration) and the nouns they use. */
export const texts = config.texts;
export const nouns = config.nouns;
/** A text with its {fields} filled in; fields the caller does not give stay as they are. */
export const fill = (key, fields = {}) => texts[key].replace(/\{(\w+)\}/g, (match, name) => (name in fields ? fields[name] : match));
