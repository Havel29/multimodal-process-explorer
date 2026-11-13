// Help: a guided tour. A spotlight glides from one part of the explorer to the next, with a short
// card explaining it. Arrow keys or the buttons move through the steps; Escape closes the tour.
import { imageNoun } from "./config.js";
import { showCaseView } from "./controller.js";
import { state } from "./state.js";

const button = document.getElementById("helpButton");
const PADDING = 6;
const GAP = 12;

// Steps whose element is missing or hidden (e.g. no Statistics without labels) are skipped.
const STEPS = [
  { target: ".view-tabs", title: "Three pages",
    text: "Inputs loads another event log, images and models, Case view explores one case at a time, and Statistics summarises every case." },
  { target: ".patient-picker", title: "Pick a case",
    text: "Choose a case from the list, or step through them with the arrows." },
  { target: "#swimlanes .lane-scroll", title: "The timeline",
    text: "One lane per event type, one mark per event. Click a lane name to leave those events out; drag a box over the lanes to pick events by time." },
  { target: "#swimlanes .cutoff-line", title: "The cutoff", demo: "cutoff",
    text: "The black line is the cutoff: the models only see what happened before it. Drag it, click the timeline, or pick a day below." },
  { target: "[data-jump-end]", title: "Jump to end",
    text: "Puts the cutoff back at the last event, with every event given to the models. A case opens this way." },
  { target: "#swimlanes .chart", title: "Prediction lines",
    text: "The probability each model gives the target class (top right) at every cutoff along the case, computed in the background after the predictions." },
  { target: "#swimlanes .overview", title: "Zoom and browse",
    text: "When events crowd together, zoom in with + (or Ctrl + scroll on the timeline). This bar is the whole case: drag its box to move along it, and double-click it to see everything again." },
  { target: "#splitter", title: "Resize", demo: "splitter",
    text: "Drag this bar down for a taller prediction chart, or up to give the panels below more room. Double-click resets it." },
  { target: ".stage-image", title: `The ${imageNoun}`,
    text: `The latest ${imageNoun} known at the cutoff, which the image models read. After an explanation, a heatmap shows which patches mattered.` },
  { target: ".knowledge", title: "Events given to the models",
    text: "Every event known at the cutoff. Untick events to see how the predictions change without them, then run again. The search box filters the list; Select all and Clear then act on the matches." },
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

async function start() {
  if (state.view !== "patient") await showCaseView();
  steps = STEPS.filter((step) => visible(document.querySelector(step.target)));
  if (!steps.length) return;
  if (!layer) build();
  document.addEventListener("keydown", onKey, true);
  window.addEventListener("resize", reposition);
  go(0);
}

export function init() {
  button.addEventListener("click", start);
}
