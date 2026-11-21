const toast = document.getElementById("toast");
let timer = null;

/** A short message; errors by default, "success" for good news. */
export function showToast(message, tone = "error") {
  toast.textContent = message;
  toast.classList.toggle("success", tone === "success");
  toast.classList.remove("hidden");
  window.clearTimeout(timer);
  timer = window.setTimeout(() => toast.classList.add("hidden"), 6000);
}
