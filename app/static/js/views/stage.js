// The image used by the image models, with the optional patch heatmap.
import { capitalized, displayName, imageNoun, riskRgb, safeRgb, texts } from "../config.js";
import { api } from "../api.js";
import { timeLabel } from "../format.js";
import { eventById, imageEvent, state } from "../state.js";

const HEATMAP_MAX_ALPHA = 205;

const els = {
  title: document.getElementById("stageTitle"),
  time: document.getElementById("stageTime"),
  frame: document.getElementById("imageFrame"),
  image: document.getElementById("stageImage"),
  empty: document.getElementById("imageEmpty"),
  heatmap: document.getElementById("heatmap"),
  toggle: document.getElementById("heatmapToggle"),
};

export function init() {
  els.image.addEventListener("load", drawHeatmap);
  els.toggle.addEventListener("change", drawHeatmap);
  // The frame also changes size without a window resize (panels opening, the layout settling)
  new ResizeObserver(drawHeatmap).observe(els.frame);
}

export function render() {
  const cutoff = state.cutoff;
  els.title.textContent = !cutoff ? "No cutoff selected"
    : cutoff.kind === "day" ? `Knowledge through ${texts.day} ${cutoff.day}`
      : cutoff.kind === "stage" ? displayName(eventById(cutoff.stageId))
        : `Knowledge up to ${timeLabel(cutoff.seconds)}`;

  const image = imageEvent();
  if (image) {
    els.time.textContent = `${capitalized(imageNoun)} from ${timeLabel(image.offset_seconds)}`;
    const source = api.imageUrl(state.patient.id, image.id);
    if (els.image.getAttribute("src") !== source) els.image.src = source;
    els.image.classList.remove("hidden");
    els.empty.classList.add("hidden");
  } else {
    els.time.textContent = cutoff ? `No ${imageNoun} yet` : "";
    els.image.removeAttribute("src");
    els.image.classList.add("hidden");
    els.empty.classList.remove("hidden");
    els.empty.textContent = cutoff ? `No ${imageNoun} available at this cutoff.` : texts.start_hint;
  }
  drawHeatmap();
}

/** Patch attributions: colour shows the direction, opacity the size. */
export function drawHeatmap() {
  const explanation = state.explanation;
  const image = imageEvent();
  const matches = Boolean(explanation && image && explanation.image_event_id === image.id);
  els.toggle.classList.toggle("hidden", !matches);
  if (!matches || !els.image.naturalWidth || !els.toggle.querySelector("input").checked) {
    els.heatmap.classList.add("hidden");
    return;
  }

  // Cover exactly the part of the image box that object-fit: contain fills
  const frame = els.frame.getBoundingClientRect();
  const box = els.image.getBoundingClientRect();
  const scale = Math.min(box.width / els.image.naturalWidth, box.height / els.image.naturalHeight);
  const width = els.image.naturalWidth * scale;
  const height = els.image.naturalHeight * scale;
  Object.assign(els.heatmap.style, {
    width: `${width}px`, height: `${height}px`,
    left: `${box.left - frame.left + (box.width - width) / 2}px`,
    top: `${box.top - frame.top + (box.height - height) / 2}px`,
  });

  const values = explanation.patches.flat();
  const size = explanation.patches.length;
  const largest = Math.max(...values.map(Math.abs)) || 1;
  els.heatmap.width = size;
  els.heatmap.height = size;
  const context = els.heatmap.getContext("2d");
  const pixels = context.createImageData(size, size);
  values.forEach((value, index) => {
    const alpha = Math.round(Math.sqrt(Math.abs(value) / largest) * HEATMAP_MAX_ALPHA);
    pixels.data.set([...(value >= 0 ? riskRgb : safeRgb), alpha], index * 4);
  });
  context.putImageData(pixels, 0, 0);
  els.heatmap.classList.remove("hidden");
}
