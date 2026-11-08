const toast = document.getElementById("toast");
let timer = null;

export function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.clearTimeout(timer);
  timer = window.setTimeout(() => toast.classList.add("hidden"), 6000);
}
