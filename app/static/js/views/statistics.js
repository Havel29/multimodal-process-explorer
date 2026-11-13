// Statistics over the loaded cases.
import { api } from "../api.js";
import { config, imageNoun, models, riskLabel, stageNoun, texts, fill } from "../config.js";
import { selectPatient } from "../controller.js";
import { escapeHtml, percent, plural, timeLabel } from "../format.js";
import { accuracy, agreement, atDay, atSharedImages, auc, brier, flips, isCorrect, macroF1, pairs, predictedClass, series, settlesFrom } from "../metrics.js";
import { state } from "../state.js";
import { showToast } from "./toast.js";

const TEXT_MODELS = models.filter((model) => model.usesText);
const TOP_TYPES = 12;
const MAX_TABLE_DAY_COLUMNS = 10;
const MIN_CHART_PATIENT_SHARE = 0.25;
let lastPatients = [];
let computeRun = 0;  // a reset bumps it, which stops a running computation
let showAllTypes = false;

const els = {
  computed: document.getElementById("statisticsComputed"),
  caveat: document.getElementById("statisticsCaveat"),
  charts: document.getElementById("statisticsCharts"),
  table: document.getElementById("statisticsTable"),
  computeMissing: document.getElementById("computeMissing"),
  reset: document.getElementById("resetStatistics"),
};

export function init() {
  els.table.addEventListener("click", (event) => {
    const open = event.target.closest("[data-open]");
    if (open) selectPatient(open.dataset.open);
  });
  els.computeMissing.addEventListener("click", computeMissing);
  els.reset.addEventListener("click", resetStatistics);
  els.charts.addEventListener("click", (event) => {
    if (!event.target.closest("[data-show-all-types]")) return;
    showAllTypes = true;
    render(lastPatients);
  });
}

/** Fill the Statistics page (the controller shows it). */
export async function show() {
  try {
    render((await api.statistics()).patients);
  } catch (error) {
    showToast(error.message);
  }
}

function stopComputing() {
  els.computeMissing.disabled = false;
  els.computeMissing.textContent = "Compute missing cases";
}

/**
 * Two passes, so the counters match what is on screen: first the risk curves (the charts),
 * then what drives the predictions. Each counts on from the cases that already have that part.
 */
async function computeMissing() {
  const run = ++computeRun;
  els.computeMissing.disabled = true;
  try {
    const { patients } = await api.statistics();
    const total = patients.length;
    const passes = [
      { label: "curves", has: (patient) => patient.points, compute: (id) => api.trajectory(id, true) },
      { label: "drivers", has: (patient) => patient.importance, compute: (id) => api.importance(id, true) },
    ];
    for (const pass of passes) {
      const missing = patients.filter((patient) => !pass.has(patient)).map((patient) => patient.id);
      const done = total - missing.length;
      for (const [index, patientId] of missing.entries()) {
        if (run !== computeRun) return;
        els.computeMissing.textContent = `Computing ${pass.label} ${done + index + 1} of ${total}`;
        await pass.compute(patientId);
        if (run !== computeRun) return;
        const latest = await api.statistics();
        if (run !== computeRun) return;
        if (state.view === "statistics") render(latest.patients);
      }
    }
  } catch (error) {
    if (run === computeRun) showToast(error.message);
  } finally {
    if (run === computeRun) stopComputing();
  }
}

/** Delete the computed statistics, so they can be computed again live. */
async function resetStatistics() {
  if (!window.confirm("Delete the computed statistics of every case? Compute missing cases computes them again.")) return;
  computeRun += 1;
  stopComputing();
  els.reset.disabled = true;
  try {
    await api.resetStatistics();
    render((await api.statistics()).patients);
    showToast("Statistics reset. Use Compute missing cases to compute them again.", "success");
  } catch (error) {
    showToast(error.message);
  } finally {
    els.reset.disabled = false;
  }
}

const maxObservedDay = (patients) => Math.max(1, ...patients.flatMap((patient) =>
  (patient.points || []).filter((point) => point.kind === "day").map((point) => point.day)));
function spacedDays(lastDay) {
  if (lastDay <= MAX_TABLE_DAY_COLUMNS) return Array.from({ length: lastDay }, (_, index) => index + 1);
  return [...new Set(Array.from({ length: MAX_TABLE_DAY_COLUMNS }, (_, index) =>
    Math.round(1 + (index * (lastDay - 1)) / (MAX_TABLE_DAY_COLUMNS - 1))))];
}
const tableDays = (patients) => spacedDays(maxObservedDay(patients));
function supportedHorizon(patients, minimum) {
  const endings = patients.map((patient) => maxObservedDay([patient])).sort((a, b) => b - a);
  return endings[Math.min(endings.length, minimum) - 1] || 1;
}
const pointLabel = (point) => (point.kind === "day" ? `end of ${texts.day} ${point.day}` : `${stageNoun}, ${timeLabel(point.offset_seconds)}`);

function render(rawPatients) {
  lastPatients = rawPatients;
  const patients = rawPatients.map((patient) => ({ ...patient, isRisk: patient.ground_truth.code === config.riskClass }));
  const computed = patients.filter((patient) => patient.points);
  const minimumChartPatients = Math.max(2, Math.ceil(computed.length * MIN_CHART_PATIENT_SHARE));
  const chartDays = spacedDays(supportedHorizon(computed, minimumChartPatients));
  const shownDays = tableDays(computed);
  // Cases opened in the Case view compute their prediction lines, and computed results are kept
  // until a reset: say so, or results present on the first visit look unexplained.
  els.computed.textContent = `${computed.length} of ${plural(patients.length, "case")} computed. `
    + "Opening a case in the Case view computes its prediction lines, so the cases you opened are already included; "
    + "results are kept until you reset them.";
  els.caveat.textContent = fill("statistics_caveat", {
    count: patients.length, last_day: `${texts.day} ${chartDays.at(-1)}`, minimum: minimumChartPatients, table_day: `${texts.day} ${shownDays.at(-1)}`,
  });
  els.charts.innerHTML = computed.length ? `
    <div class="metric-charts">
      ${lineChart("Accuracy", "Share of cases whose outcome is predicted correctly.", byDay(computed, accuracy, chartDays), chartDays, minimumChartPatients)}
      ${lineChart("ROC AUC", "Chance that a case with the risk outcome receives a higher risk than one without (0.5 is random).", byDay(computed, auc, chartDays), chartDays, minimumChartPatients)}
      ${lineChart("Brier score", "Mean squared gap between the predicted probability and the outcome (1 or 0). 0 is perfect; always answering 50% scores 0.25. Lower is better.", byDay(computed, brier, chartDays), chartDays, minimumChartPatients)}
    </div>
    <div class="statistics-panels">
      <section><h2>Model summary</h2>${summaryTable(computed)}</section>
      <section><h2>What drives the predictions</h2>${importance(patients)}</section>
    </div>` : "";
  els.table.innerHTML = patientTable(patients, shownDays);

  const missing = patients.filter((patient) => !patient.points || !patient.importance).map((patient) => patient.id);
  els.computeMissing.classList.toggle("hidden", !missing.length);
  els.reset.classList.toggle("hidden", !patients.some((patient) => patient.points || patient.importance));
}

function byDay(patients, metric, dayGrid) {
  return Object.fromEntries(models.map((model) => [model.id, dayGrid.map((day) => {
    const samples = pairs(patients, model.id, atDay(day));
    return { value: metric(samples), count: samples.length };
  })]));
}

/** A small chart over Day 1 to the last cutoff day, one line per model, labelled at the end. */
function lineChart(title, explanation, values, dayGrid, minimumPatients) {
  const width = 320;
  const height = 170;
  const pad = { left: 30, right: 12, top: 12, bottom: 26 };
  const lastDay = dayGrid.at(-1) || 1;
  const x = (day) => pad.left + ((day - 1) / Math.max(1, lastDay - 1)) * (width - pad.left - pad.right);
  const y = (value) => pad.top + (1 - value) * (height - pad.top - pad.bottom);
  const grid = [0, 0.5, 1].map((tick) => `<line class="grid" x1="${pad.left}" x2="${width - pad.right}" y1="${y(tick)}" y2="${y(tick)}"/>
    <text class="tick" x="${pad.left - 6}" y="${y(tick) + 4}" text-anchor="end">${tick}</text>`).join("");
  const ticks = [...new Set([1, Math.round(lastDay / 3), Math.round((2 * lastDay) / 3), lastDay])]
    .map((day) => `<text class="tick" x="${x(day)}" y="${height - 8}" text-anchor="middle">${texts.day} ${day}</text>`).join("");

  const lines = models.map((model) => {
    const defined = values[model.id].map((entry, index) => ({ day: dayGrid[index], ...entry }))
      .filter((item) => item.value !== null && item.count >= minimumPatients);
    if (!defined.length) return "";
    const dots = defined.map((item) => `<circle cx="${x(item.day)}" cy="${y(item.value)}" r="3.5" style="fill: var(--model-${model.id})">
      <title>${escapeHtml(model.name)}, ${texts.day} ${item.day}: ${item.value.toFixed(2)} (n=${item.count})</title></circle>`).join("");
    return `<polyline points="${defined.map((item) => `${x(item.day)},${y(item.value)}`).join(" ")}" style="stroke: var(--model-${model.id})"/>${dots}`;
  }).join("");
  const legend = models.map((model) => `<span><i class="swatch" style="background: var(--model-${model.id})"></i>${escapeHtml(model.name)}</span>`).join("");
  return `<figure class="metric-chart"><figcaption><strong>${escapeHtml(title)}</strong><span>${escapeHtml(explanation)}</span></figcaption>
    <div class="chart-legend">${legend}</div>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(title)} by day">${grid}${ticks}${lines}</svg></figure>`;
}

function summaryTable(patients) {
  const fixed = (value, digits) => (value === null ? "-" : value.toFixed(digits));
  const imageCount = pairs(patients, models[0]?.id, atSharedImages).length;
  const rows = models.map((model) => {
    const all = pairs(patients, model.id);
    const images = pairs(patients, model.id, atSharedImages);
    const settled = patients.map((patient) => settlesFrom(patient.points, model.id, patient.ground_truth.code)).filter(Boolean);
    const meanDays = settled.length ? settled.reduce((sum, point) => sum + point.offset_seconds, 0) / settled.length / 86400 : null;
    const meanFlips = patients.reduce((sum, patient) => sum + flips(patient.points, model.id), 0) / patients.length;
    return `<tr><td class="left"><i class="swatch" style="background: var(--model-${model.id})"></i>${escapeHtml(model.name)}</td>
      <td>${images.length ? percent(accuracy(images)) : "-"}</td><td>${fixed(auc(images), 2)}</td><td>${fixed(macroF1(images), 2)}</td>
      <td class="group-start">${all.length ? percent(accuracy(all)) : "-"}</td><td>${fixed(auc(all), 2)}</td><td>${fixed(brier(all), 3)}</td>
      <td class="group-start">${settled.length} of ${patients.length}</td><td>${meanDays === null ? "-" : `${meanDays.toFixed(1)} days`}</td>
      <td>${meanFlips.toFixed(1)}</td></tr>`;
  }).join("");
  const agreements = patients.map((patient) => agreement(patient.points)).filter(Boolean);
  const agreeShare = agreements.length
    ? percent(agreements.reduce((sum, item) => sum + item.agree, 0) / agreements.reduce((sum, item) => sum + item.total, 0))
    : "-";
  return `<table class="summary-table"><thead>
    <tr><th></th><th colspan="3">At each ${escapeHtml(imageNoun)} (${imageCount})</th><th colspan="3" class="group-start">All cutoffs</th><th colspan="3" class="group-start">Along each ${escapeHtml(config.nouns?.pathway || "case")}</th></tr>
    <tr><th class="left">Model</th><th>Accuracy</th><th>AUC</th><th>Macro F1</th>
    <th class="group-start">Accuracy</th><th>AUC</th><th>Brier</th>
    <th class="group-start">Settles on outcome</th><th>Mean time to settle</th><th>Flips per case</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="note">${escapeHtml(texts.summary_note)}${models.length > 1
      ? ` All models agree at ${agreeShare} of the cutoffs where every model can run.` : ""}</p>`;
}

function importance(patients) {
  const profiles = patients.flatMap((patient) => patient.importance || []);
  if (!profiles.length) return '<p class="empty-note">Event-type importance has not been computed yet.</p>';
  const totals = new Map();
  for (const profile of profiles) {
    for (const row of profile.rows) {
      const entry = totals.get(row.key) || { label: row.label, models: {} };
      for (const [model, delta] of Object.entries(row.delta)) {
        const stats = entry.models[model] || (entry.models[model] = { absolute: 0, signed: 0, count: 0 });
        stats.absolute += Math.abs(delta);
        stats.signed += delta;
        stats.count += 1;
      }
      totals.set(row.key, entry);
    }
  }
  const mean = (stats) => stats.absolute / stats.count;
  const rows = [...totals.values()].sort((a, b) =>
    Math.max(...Object.values(b.models).map(mean)) - Math.max(...Object.values(a.models).map(mean)));
  const largest = Math.max(0.01, ...rows.flatMap((row) => Object.values(row.models).map(mean)));
  const shown = showAllTypes ? rows : rows.slice(0, TOP_TYPES);
  const bars = shown.map((row) => `<div class="importance-row"><span>${escapeHtml(row.label)}</span><div>${TEXT_MODELS
    .filter((model) => row.models[model.id]).map((model) => {
      const stats = row.models[model.id];
      const signed = stats.signed / stats.count;
      return `<div class="importance-bar" title="${escapeHtml(`${model.name}: mean change ${(mean(stats) * 100).toFixed(1)} points in size, ${(signed * 100).toFixed(1)} on average, over ${plural(stats.count, `${imageNoun} cutoff`)}`)}">
        <i class="swatch" style="background: var(--model-${model.id})"></i>
        <span class="bar-track"><span style="width: ${(mean(stats) / largest) * 100}%; background: var(--model-${model.id})"></span></span>
        <em>${(mean(stats) * 100).toFixed(1)}</em></div>`;
    }).join("")}</div></div>`).join("");
  const hidden = rows.length - shown.length;
  return `<div class="importance">${bars}</div>
    ${hidden ? `<button class="show-all" data-show-all-types type="button">Show all ${rows.length} event types (${hidden} with smaller effects hidden)</button>` : ""}
    <p class="note">${escapeHtml(fill("importance_note", { risk: riskLabel, count: profiles.length }))}</p>`;
}

/** One day of the per-case table: the risk probability, coloured by the predicted class. */
function cell(point, model, truth) {
  const value = point?.risk[model];
  if (value === null || value === undefined) return '<td><span class="cell empty">-</span></td>';
  const predicted = predictedClass(point, model);
  const strength = 0.12 + Math.abs(value - 0.5) * 1.1;
  const tone = predicted === config.riskClass ? "var(--risk)" : `var(--outcome-${predicted})`;
  const miss = predicted !== truth;
  const label = config.outcomes.find((outcome) => outcome.code === predicted)?.label || predicted;
  return `<td><span class="cell ${miss ? "miss" : ""} ${strength > 0.45 ? "strong" : ""}" style="--tone: ${tone}; --strength: ${strength}"
    title="${escapeHtml(`${riskLabel} ${percent(value)}; predicts ${label}${miss ? ", differs from the observed outcome" : ""}`)}">${Math.round(value * 100)}</span></td>`;
}

function patientTable(patients, dayGrid) {
  const header = `<tr><th class="left">Case</th><th class="left">Observed</th><th class="left">Model</th>
    ${dayGrid.map((day) => `<th>${texts.day} ${day}</th>`).join("")}<th>At ${stageNoun}s</th><th class="left">Settles from</th></tr>`;
  const body = patients.map((patient) => {
    if (!patient.points) {
      return `<tr class="patient-first"><td class="left"><strong>${escapeHtml(patient.id)}</strong></td>
        <td class="left">${escapeHtml(patient.ground_truth.label)}</td>
        <td class="left muted" colspan="${dayGrid.length + 3}">Not computed yet</td></tr>`;
    }
    return models.map((model, index) => {
      const cells = dayGrid.map((day) => cell(atDay(day)(patient.points)[0], model.id, patient.ground_truth.code)).join("");
      const stages = series(patient.points.filter((point) => point.kind === "stage"), model.id);
      const right = stages.filter((point) => isCorrect(point, model.id, patient.ground_truth.code)).length;
      const runs = series(patient.points, model.id).length > 0;
      const settled = runs ? settlesFrom(patient.points, model.id, patient.ground_truth.code) : null;
      const first = index === 0 ? `
        <td class="left" rowspan="${models.length}"><strong>${escapeHtml(patient.id)}</strong><br><button class="open" data-open="${escapeHtml(patient.id)}" type="button">Open</button></td>
        <td class="left" rowspan="${models.length}" style="color: var(--outcome-${patient.ground_truth.code})">${escapeHtml(patient.ground_truth.label)}</td>` : "";
      return `<tr class="${index === 0 ? "patient-first" : ""}">${first}
        <td class="left"><i class="swatch" style="background: var(--model-${model.id})"></i>${escapeHtml(model.name)}</td>${cells}
        <td>${stages.length ? `${right} of ${stages.length}` : "-"}</td>
        <td class="left">${!runs ? "-" : settled ? escapeHtml(pointLabel(settled)) : '<span class="never">never</span>'}</td></tr>`;
    }).join("");
  }).join("");
  return `<h2>Per-case predictions</h2>
    <p class="note">Cells show ${escapeHtml(riskLabel)} at ${dayGrid.length} evenly spaced checkpoints from ${texts.day} ${dayGrid[0]} to
    ${texts.day} ${dayGrid.at(-1)}, coloured by the predicted class; "-" means that trace ended earlier. A yellow corner marks a prediction
    that differs from the observed outcome. "Settles from" still considers every cutoff across the whole stay.</p>
    <table><thead>${header}</thead><tbody>${body}</tbody></table>`;
}
