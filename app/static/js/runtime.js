import { api } from "./api.js";
import { state } from "./state.js";
import * as predictions from "./views/predictions.js";
import { showToast } from "./views/toast.js";

const selector = document.getElementById("deviceSelect");

export async function init() {
  const runtime = await api.runtime();
  selector.value = runtime.preference;
  const gpu = selector.querySelector('option[value="gpu"]');
  gpu.disabled = !runtime.gpu_available;
  // The GPU name goes in the tooltip: a long name would widen the header
  gpu.textContent = runtime.gpu_available ? "GPU" : "GPU (unavailable)";
  if (runtime.gpu_available) selector.title = `GPU: ${runtime.gpu_name}`;

  selector.addEventListener("change", async () => {
    const previous = runtime.preference;
    selector.disabled = true;
    try {
      const changed = await api.setDevice(selector.value);
      runtime.preference = changed.preference;
      state.runToken += 1;
      state.predicting = false;
      predictions.reset();
      showToast(`Inference device set to ${changed.preference.toUpperCase()}. Models will reload on the next run.`, "success");
    } catch (error) {
      selector.value = previous;
      showToast(error.message);
    } finally {
      selector.disabled = false;
    }
  });
}
