// Entry point: the Inputs page, or the explorer (#explorer, #statistics) on the data already loaded.
import { api } from "./api.js";
import { selectPatient, showCaseView, showStatistics } from "./controller.js";
import { state } from "./state.js";
import * as analysis from "./views/analysis.js";
import * as header from "./views/header.js";
import * as knowledge from "./views/knowledge.js";
import * as predictions from "./views/predictions.js";
import * as stage from "./views/stage.js";
import * as statistics from "./views/statistics.js";
import * as timeline from "./views/timeline.js";
import * as splitter from "./views/splitter.js";
import { showToast } from "./views/toast.js";
import * as setup from "./setup.js";
import * as runtime from "./runtime.js";
import * as exportCsv from "./export.js";
import * as tour from "./tour.js";

try {
  const status = await api.setupStatus();
  const page = window.location.hash.slice(1);
  const exploring = status.configured && [setup.EXPLORER_HASH, setup.STATISTICS_HASH].includes(page);
  setup.init({ open: !exploring, canContinue: status.configured, onContinue: exploring ? showCaseView : null });
  if (exploring) {
    for (const view of [header, timeline, stage, knowledge, predictions, analysis, statistics, exportCsv, splitter, tour]) view.init();
    await runtime.init();
    state.patients = (await api.patients()).patients;
    header.renderTabs();
    if (state.patients.length) await selectPatient(state.patients[0].id);
    if (page === setup.STATISTICS_HASH && !document.getElementById("statisticsTab").classList.contains("hidden")) await showStatistics();
  } else {
    header.initSetupPage(status.configured, setup.enterExplorer);
  }
} catch (error) {
  document.getElementById("patientTitle").textContent = "Input data is not prepared yet";
  showToast(error.message);
}
