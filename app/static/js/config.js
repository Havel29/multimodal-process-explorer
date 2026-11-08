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
export const riskOutcome = config.outcomes.find((outcome) => outcome.code === config.riskClass) || PLACEHOLDER_RISK;
export const otherOutcome = config.outcomes.find((outcome) => outcome.code !== config.riskClass) || PLACEHOLDER_OTHER;
export const riskLabel = `P(${riskOutcome.short})`;

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

export const riskRgb = rgb(riskOutcome.color);
export const safeRgb = rgb(otherOutcome.color);

/** The texts of the interface ([texts] in the configuration) and the nouns they use. */
export const texts = config.texts;
export const nouns = config.nouns;
/** A text with its {fields} filled in; fields the caller does not give stay as they are. */
export const fill = (key, fields = {}) => texts[key].replace(/\{(\w+)\}/g, (match, name) => (name in fields ? fields[name] : match));
