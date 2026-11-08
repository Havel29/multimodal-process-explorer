import { api } from "./api.js";

const wizard = document.getElementById("setupWizard");
const pathsForm = document.getElementById("setupPaths");
const fieldsForm = document.getElementById("setupFields");
const message = document.getElementById("setupMessage");
const step = document.getElementById("setupStep");
const intro = document.getElementById("setupIntro");
const openButton = document.getElementById("setupOpen");
const continueButton = document.getElementById("setupContinue");
let token = null;
let listenersBound = false;

export const EXPLORER_HASH = "explorer";

function enterExplorer() {
  window.location.hash = EXPLORER_HASH;
  window.location.reload();
}

/** What the setup read from the log instead of the configuration, to check before going on. */
function showNotes(notes) {
  busy(fieldsForm, "");
  message.replaceChildren();
  const intro = document.createElement("p");
  intro.textContent = "Ready. The configuration leaves these out, so they were read from the event log; "
    + "if they look right, there is nothing to do:";
  const list = document.createElement("ul");
  for (const note of notes) {
    const item = document.createElement("li");
    item.textContent = note;
    list.append(item);
  }
  const button = document.createElement("button");
  button.type = "button";
  button.className = "primary-button";
  button.textContent = "Open the explorer";
  button.addEventListener("click", enterExplorer);
  message.append(intro, list, button);
  // The setup is already applied: the top button would only parse the data again. It comes back if a field changes.
  const submit = fieldsForm.querySelector("button[type=submit]");
  submit.disabled = true;
  fieldsForm.addEventListener("change", () => {
    submit.disabled = false;
    message.replaceChildren();
  }, { once: true });
}

const fieldSelects = {
  case_field: document.getElementById("caseField"),
  activity_field: document.getElementById("activityField"),
  timestamp_field: document.getElementById("timestampField"),
  image_field: document.getElementById("imageField"),
  label_field: document.getElementById("labelField"),
};

const modelSelects = {
  text_model: document.getElementById("textModel"),
  image_model: document.getElementById("imageModel"),
  multimodal_model: document.getElementById("multimodalModel"),
};

const MODEL_EXTENSIONS = [".pth", ".pt", ".bin", ".ckpt", ".safetensors"];

function fill(select, values, preferred = [], empty = "Select a value") {
  select.replaceChildren();
  select.add(new Option(empty, "", true, true));
  for (const value of values) select.add(new Option(value, value));
  const candidates = preferred.map((name) => name.toLowerCase());
  // Preference order matters: an exact configured field must win over a
  // broader fallback such as "concept:name" appearing inside another column.
  let match = candidates.map((name) => values.find((value) => value.toLowerCase() === name)).find(Boolean);
  if (!match) match = candidates.map((name) => values.find((value) => value.toLowerCase().includes(name))).find(Boolean);
  if (match) select.value = match;
}

function busy(form, text) {
  for (const element of form.elements) element.disabled = Boolean(text);
  message.textContent = text || "";
}

function relativePath(file) {
  const parts = (file.webkitRelativePath || file.name).split("/").filter(Boolean);
  return parts.length > 1 ? parts.slice(1).join("/") : parts[0];
}

async function uploadFiles(uploadToken, kind, files, concurrency) {
  let completed = 0;
  for (let offset = 0; offset < files.length; offset += concurrency) {
    const batch = files.slice(offset, offset + concurrency);
    await Promise.all(batch.map((file) => api.uploadSetupFile(uploadToken, kind, relativePath(file), file)));
    completed += batch.length;
    busy(pathsForm, `Uploading ${kind} files: ${completed} of ${files.length}...`);
  }
}

function reset() {
  token = null;
  pathsForm.reset();
  fieldsForm.reset();
  for (const select of [...Object.values(fieldSelects), ...Object.values(modelSelects)]) select.replaceChildren();
  pathsForm.classList.remove("hidden");
  fieldsForm.classList.add("hidden");
  intro.classList.remove("hidden");
  step.textContent = "Multimodal setup - Step 1 of 2";
  message.textContent = "";
  busy(pathsForm, "");
  busy(fieldsForm, "");
}

function open() {
  reset();
  wizard.classList.remove("hidden");
  document.getElementById("xesFile").focus();
}

function bindListeners() {
  if (listenersBound) return;
  listenersBound = true;
  openButton.addEventListener("click", open);
  continueButton.addEventListener("click", enterExplorer);

  pathsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const xes = document.getElementById("xesFile").files[0];
    const images = [...document.getElementById("imageFiles").files];
    const models = [...document.getElementById("modelFiles").files].filter((file) =>
      MODEL_EXTENSIONS.some((extension) => file.name.toLowerCase().endsWith(extension)));
    busy(pathsForm, "Preparing folder upload...");
    try {
      if (!xes || !images.length) throw new Error("Select the XES file and image folder");
      if (!models.length) throw new Error("The selected model folder contains no supported checkpoints");
      const upload = await api.createSetupUpload();
      await uploadFiles(upload.token, "images", images, 4);
      await uploadFiles(upload.token, "models", models, 1);
      busy(pathsForm, "Parsing XES and inspecting model files...");
      const result = await api.inspectStagedSetup(upload.token, xes);
      token = result.token;
      fill(fieldSelects.case_field, result.fields, [result.field_defaults.case, "case:concept:name", "patient_id", "case_id"].filter(Boolean));
      fill(fieldSelects.activity_field, result.fields, [result.field_defaults.activity, "concept:name", "activity"].filter(Boolean));
      fill(fieldSelects.timestamp_field, result.fields, [result.field_defaults.timestamp, "time:timestamp", "timestamp"].filter(Boolean));
      fill(fieldSelects.image_field, result.fields, [result.field_defaults.image, "path", "image", "image_path"].filter(Boolean));
      fill(fieldSelects.label_field, result.fields, [result.field_defaults.outcome, "outcome", "label"].filter(Boolean), "No label");
      // One field per model role of the configuration; a field left on "Not used" leaves that model out
      const guesses = { text: ["textonly", "text"], image: ["imageonly", "image"], multimodal: ["multimodal", "multi"] };
      for (const [role, guess] of Object.entries(guesses)) {
        const select = modelSelects[`${role}_model`];
        const configured = role in result.model_defaults;
        select.closest("label").classList.toggle("hidden", !configured);
        if (configured) fill(select, result.model_files, [result.model_defaults[role], ...guess].filter(Boolean), "Not used");
      }
      document.getElementById("setupSummary").textContent = `${result.event_count} events parsed. Map the XES fields and checkpoints.`;
      busy(pathsForm, "");
      pathsForm.classList.add("hidden");
      fieldsForm.classList.remove("hidden");
      intro.classList.add("hidden");
      step.textContent = "Multimodal setup - Step 2 of 2";
      message.textContent = "";
    } catch (error) {
      busy(pathsForm, "");
      message.textContent = error.message;
    }
  });

  document.getElementById("setupBack").addEventListener("click", () => {
    fieldsForm.classList.add("hidden");
    pathsForm.classList.remove("hidden");
    intro.classList.remove("hidden");
    step.textContent = "Multimodal setup - Step 1 of 2";
    message.textContent = "";
  });

  fieldsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    busy(fieldsForm, "Preparing the cases and configuring the models...");
    try {
      const body = { token };
      for (const [name, select] of Object.entries(fieldSelects)) body[name] = select.value || null;
      for (const [name, select] of Object.entries(modelSelects)) body[name] = select.value || null;
      if (!Object.values(modelSelects).some((select) => select.value)) throw new Error("Select at least one model");
      const result = await api.applySetup(body);
      if (result.notes?.length) return showNotes(result.notes);
      enterExplorer();
    } catch (error) {
      busy(fieldsForm, "");
      message.textContent = error.message;
    }
  });
}

export function init({ open: openAtStart = true, canContinue = false } = {}) {
  bindListeners();
  continueButton.classList.toggle("hidden", !canContinue);
  if (openAtStart) open();
}
