// A bar centred on zero: right (risk colour) for positive values, left otherwise.
export function divergingBar(value, largest) {
  const width = Math.min(50, (Math.abs(value) / (largest || 1)) * 50);
  return `<span class="diverging"><i class="${value >= 0 ? "toward-risk" : "away-from-risk"}" style="width: ${width}%"></i></span>`;
}
