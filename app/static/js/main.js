// Entry point: wire the views, load the cases, and open the first one.
import { api } from "./api.js";
import { selectPatient } from "./controller.js";
import { state } from "./state.js";
import * as analysis from "./views/analysis.js";
import * as header from "./views/header.js";
import * as knowledge from "./views/knowledge.js";
import * as predictions from "./views/predictions.js";
import * as stage from "./views/stage.js";
import * as statistics from "./views/statistics.js";
import * as timeline from "./views/timeline.js";
import { showToast } from "./views/toast.js";
import * as setup from "./setup.js";
import * as runtime from "./runtime.js";
import * as exportCsv from "./export.js";

try {
  const status = await api.setupStatus();
  // The inputs page comes first; the dashboard needs the #explorer address.
  const exploring = status.configured && window.location.hash === `#${setup.EXPLORER_HASH}`;
  setup.init({ open: !exploring, canContinue: status.configured });
  if (exploring) {
    for (const view of [header, timeline, stage, knowledge, predictions, analysis, statistics, exportCsv]) view.init();
    await runtime.init();
    state.patients = (await api.patients()).patients;
    header.renderTabs();
    if (state.patients.length) await selectPatient(state.patients[0].id);
  }
} catch (error) {
  document.getElementById("patientTitle").textContent = "Input data is not prepared yet";
  showToast(error.message);
}
