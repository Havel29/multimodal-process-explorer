// Thin client for the demo's JSON API.

async function request(url, options) {
  const response = await fetch(url, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch (_) {
    // Error pages may not be JSON.
  }
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

const patientUrl = (patientId) => `/api/patients/${encodeURIComponent(patientId)}`;

export const api = {
  setupStatus: () => request("/api/setup/status"),
  inspectSetup: (file, imageDir, modelDir) => request(`/api/setup/upload?${new URLSearchParams({
    filename: file.name, image_dir: imageDir, model_dir: modelDir,
  })}`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file }),
  createSetupUpload: () => post("/api/setup/staging", {}),
  uploadSetupFile: (token, kind, path, file) => request(`/api/setup/staging/${encodeURIComponent(token)}/file/${kind}?${new URLSearchParams({ path })}`, {
    method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
  }),
  inspectStagedSetup: (token, file) => request(`/api/setup/staging/${encodeURIComponent(token)}/inspect?${new URLSearchParams({ filename: file.name })}`, {
    method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
  }),
  applySetup: (body) => post("/api/setup/apply", body),
  runtime: () => request("/api/runtime"),
  setDevice: (device) => post("/api/runtime/device", { device }),
  performance: () => request("/api/performance"),
  patients: () => request("/api/patients"),
  patient: (patientId) => request(patientUrl(patientId)),
  imageUrl: (patientId, eventId) => `${patientUrl(patientId)}/events/${encodeURIComponent(eventId)}/image`,
  trajectory: (patientId, compute = false) => request(`${patientUrl(patientId)}/trajectory${compute ? "?compute=true" : ""}`),
  importance: (patientId, compute = false) => request(`${patientUrl(patientId)}/importance${compute ? "?compute=true" : ""}`),
  statistics: () => request("/api/statistics"),
  predict: (selection, model) => post("/api/predict", { ...selection, model }),
  explain: (selection, scope = "context", model = null) => post("/api/explain", { ...selection, scope, model }),
};

function post(url, body) {
  return request(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}
