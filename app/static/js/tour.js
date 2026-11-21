// Help: a guided tour of the page that is open (Inputs, Case view or Statistics). A spotlight glides
// from one part of the page to the next, with a short card explaining it. Arrow keys or the buttons
// move through the steps; Escape closes the tour.
import { imageNoun } from "./config.js";
import { state } from "./state.js";

const button = document.getElementById("helpButton");
const PADDING = 6;
const GAP = 12;

// Steps whose element is missing or hidden (e.g. no Statistics without labels) are skipped.
const CASE_STEPS = [
  { target: ".view-tabs", title: "Three pages",
    text: "Inputs loads another event log, images and models, Case view explores one case at a time, and Statistics summarises every case." },
  { target: ".patient-picker", title: "Pick a case",
    text: "Choose a case from the list, or step through them with the arrows." },
  { target: "#swimlanes .lane-scroll", title: "The timeline",
    text: "One lane per event type, one mark per event. Click a mark to read the event; click a lane name to leave all its events out, and again to add them back." },
  { target: "#swimlanes .lane-scroll", title: "Drag a box over events", demo: "box",
    text: "The quickest way to ask what-if questions. Press on an empty spot of the lanes and drag a box over some events, of one lane or several, then choose: Use only these (the models see just the events in the box), Exclude (they are left out) or Add back (they are given again). The predictions run again at once, so you see what those events change." },
  { target: "#swimlanes .cutoff-line", title: "The cutoff", demo: "cutoff",
    text: "The black line is the cutoff: the models only see what happened before it. Drag it, click the timeline, or pick a day below." },
  { target: "#swimlanes .axis-side", title: "Select all, Deselect all",
    text: "Both put the cutoff at the last event. Select all gives the models every event (a case opens this way); Deselect all gives none, so you can add back only the events you want." },
  { target: "#swimlanes .chart", title: "Prediction lines",
    text: "The probability each model gives the target class (top right) at every cutoff along the case, computed in the background after the predictions." },
  { target: "#swimlanes .overview", title: "Zoom and browse",
    text: "When events crowd together, zoom in with + (or Ctrl + scroll on the timeline). This bar is the whole case: drag its box to move along it, and double-click it to see everything again." },
  { target: "#splitter", title: "Resize", demo: "splitter",
    text: "Drag this bar down for a taller prediction chart, or up to give the panels below more room. Double-click resets it." },
  { target: ".stage-image", title: `The ${imageNoun}`,
    text: `The latest ${imageNoun} known at the cutoff, which the image models read. After an explanation, a heatmap shows which patches mattered.` },
  { target: ".knowledge", title: "Events given to the models",
    text: "Every event of the case; those after the cutoff are greyed out. Untick events to see how the predictions change without them, then run again. The search box filters the list; Select all and Clear then act on the matches." },
  { target: ".predictions .model-results", title: "Predictions",
    text: "Each model's prediction, with its probabilities and whether it matches the observed outcome." },
  { target: ".analysis", title: "Why?",
    text: "Explain a prediction: which events and words pushed it towards or away from the risk outcome." },
  { target: "#statisticsTab", title: "Statistics",
    text: "Accuracy, AUC and stability over every case, as knowledge accumulates." },
  { target: "#classControl", title: "Target class",
    text: "The class whose probability the lines, the statistics and the explanations show; changing it runs the predictions again. It starts at the rarest class; pick any other." },
  { target: ".runtime-controls", title: "Device, data and help",
    text: "Choose where the models run (CPU or GPU), download what the page shows as CSV, and reopen this tour." },
];

const INPUT_STEPS = [
  { target: "#setupPaths", title: "Your data",
    text: "Three inputs: an XES event log, the folder with the images its events point to, and a folder with one to three models (text, image, multimodal). Parse inputs reads them; the next step asks which attributes hold the case, activity, time, image and outcome, and which model plays each role (Help explains it there too)." },
  { target: "#setupContinue", title: "Data already loaded",
    text: "The last setup stays loaded: continue with it instead of uploading the files again." },
  { target: "#setupSummary", title: "What was found",
    text: "The cases, events and images found in your files." },
  { target: "#setupFields .setup-grid", title: "Map the fields",
    text: "Pick the XES attributes holding the case, the activity, the time and the image of each event, and the label with the outcome (needed for the Statistics). Then choose the model for each role, or Not used." },
  { target: "#setupFields .setup-actions", title: "Apply",
    text: "Apply setup prepares the cases and opens the Case view. Back returns to the files." },
  { target: ".view-tabs", title: "Three pages",
    text: "Case view and Statistics open once the data is loaded; this Inputs tab always brings you back here." },
];

const STATISTICS_STEPS = [
  { target: "#statisticsView .statistics-head", title: "Statistics",
    text: "How good the models are over every case, cutoff by cutoff, for the target class chosen at the top right." },
  { target: "#statisticsComputed", title: "Cases computed",
    text: "How many cases have their prediction lines. A case you open in the Case view is computed there and counted here." },
  { target: "#computeMissing", title: "Compute missing cases",
    text: "Runs the models on the remaining cases in the background; you can keep using the explorer meanwhile. Reset statistics starts again from scratch." },
  { target: "#statisticsView .metric-charts", title: "Day by day",
    text: "Accuracy, ROC AUC and Brier score of each model as knowledge accumulates along the cases." },
  { target: "#statisticsView .statistics-panels section:first-child", title: "Model summary",
    text: `Each model at each ${imageNoun} (the MEDUSA protocol), over all cutoffs, and along each case: how often it settles on the observed outcome, how soon, and how often it changes its mind.` },
  { target: "#statisticsView .statistics-panels section:last-child", title: "What drives the predictions",
    text: "For each event type, how much the probability of the target class changes when those events are removed." },
  { target: "#statisticsTable", title: "Per-case predictions",
    text: "Every case and model at evenly spaced days, coloured by the predicted class; a yellow corner marks a wrong prediction." },
  { target: "#classControl", title: "Target class",
    text: "The class the AUC, the Brier score and the importance refer to. Change it to read the page for another class." },
  { target: "#exportCsv", title: "Download CSV",
    text: "Every computed case: prediction lines with the probability of each class, and the event-type importance." },
];

const TOURS = { inputs: INPUT_STEPS, patient: CASE_STEPS, statistics: STATISTICS_STEPS };

let layer = null;
let index = 0;
let steps = [];

function visible(element) {
  if (!element) return false;
  const rect = element.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0 && !element.closest(".hidden");
}

function build() {
  layer = document.createElement("div");
  layer.className = "tour";
  layer.innerHTML = `<div class="tour-spot"></div>
    <div class="tour-card" role="dialog" aria-modal="true" aria-labelledby="tourTitle">
      <small class="tour-count"></small>
      <h2 id="tourTitle"></h2>
      <p class="tour-text"></p>
      <div class="tour-actions">
        <button class="tour-close secondary-button" type="button" title="Close the tour (Esc)">Skip</button>
        <span>
          <button class="tour-back secondary-button" type="button">Back</button>
          <button class="tour-next primary-button" type="button">Next</button>
        </span>
      </div>
    </div>`;
  layer.querySelector(".tour-close").addEventListener("click", close);
  layer.querySelector(".tour-back").addEventListener("click", () => go(index - 1));
  layer.querySelector(".tour-next").addEventListener("click", () => (index === steps.length - 1 ? close() : go(index + 1)));
  // A click outside the card closes the tour
  layer.addEventListener("click", (event) => {
    if (event.target === layer) close();
  });
  document.body.appendChild(layer);
}

function onKey(event) {
  if (event.key === "Escape") close();
  else if (event.key === "ArrowRight") go(Math.min(index + 1, steps.length - 1));
  else if (event.key === "ArrowLeft") go(Math.max(index - 1, 0));
}

function place(step) {
  const target = document.querySelector(step.target);
  target.scrollIntoView({ block: "nearest", inline: "nearest" });
  const rect = target.getBoundingClientRect();
  const spot = layer.querySelector(".tour-spot");
  Object.assign(spot.style, {
    left: `${rect.left - PADDING}px`, top: `${rect.top - PADDING}px`,
    width: `${rect.width + 2 * PADDING}px`, height: `${rect.height + 2 * PADDING}px`,
  });
  spot.dataset.demo = step.demo || "";

  // The card goes below the target if it fits, else above, else beside it
  const card = layer.querySelector(".tour-card");
  card.querySelector("#tourTitle").textContent = step.title;
  card.querySelector(".tour-text").textContent = step.text;
  card.querySelector(".tour-count").textContent = `${index + 1} of ${steps.length}`;
  card.querySelector(".tour-back").disabled = index === 0;
  card.querySelector(".tour-next").textContent = index === steps.length - 1 ? "Done" : "Next";
  const box = card.getBoundingClientRect();
  const below = rect.bottom + PADDING + GAP;
  const above = rect.top - PADDING - GAP - box.height;
  let top = below + box.height <= window.innerHeight - GAP ? below : above >= GAP ? above : Math.max(GAP, rect.top);
  let left = rect.left + rect.width / 2 - box.width / 2;
  if (top === Math.max(GAP, rect.top) && top !== below && top !== above) {
    left = rect.right + GAP + box.width <= window.innerWidth ? rect.right + GAP : rect.left - GAP - box.width;
  }
  left = Math.min(Math.max(GAP, left), window.innerWidth - box.width - GAP);
  top = Math.min(Math.max(GAP, top), window.innerHeight - box.height - GAP);
  Object.assign(card.style, { left: `${left}px`, top: `${top}px` });
  // Replay the entrance animation of the card
  card.classList.remove("enter");
  void card.offsetWidth;
  card.classList.add("enter");
  card.querySelector(".tour-next").focus();
}

function go(next) {
  if (!layer || next < 0 || next >= steps.length) return;
  index = next;
  place(steps[index]);
}

function close() {
  document.removeEventListener("keydown", onKey, true);
  window.removeEventListener("resize", reposition);
  layer?.remove();
  layer = null;
  button.focus();
}

function reposition() {
  if (layer) place(steps[index]);
}

/** The page open: the setup page counts as Inputs even before the explorer has loaded any data. */
function page() {
  if (visible(document.getElementById("setupWizard"))) return "inputs";
  return state.view in TOURS ? state.view : "patient";
}

function start() {
  steps = TOURS[page()].filter((step) => visible(document.querySelector(step.target)));
  if (!steps.length) return;
  if (!layer) build();
  document.addEventListener("keydown", onKey, true);
  window.addEventListener("resize", reposition);
  go(0);
}

export function init() {
  button.addEventListener("click", start);
}
