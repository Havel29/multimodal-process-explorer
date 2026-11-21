import { DAY, texts } from "./config.js";

export function escapeHtml(value) {
  const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" };
  return String(value ?? "").replace(/[&<>'"]/g, (char) => entities[char]);
}

export function clock(seconds) {
  const hours = Math.floor((seconds % DAY) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
}

/** Days are numbered from 1: the first 24 hours after admission are Day 1. */
export const dayOf = (seconds) => Math.floor(seconds / DAY) + 1;
export const timeLabel = (seconds) => `${texts.day} ${dayOf(seconds)}, ${clock(seconds)}`;
export const percent = (value) => `${(value * 100).toFixed(1)}%`;

/** A probability difference in signed percentage points. */
export const points = (value) => `${value >= 0 ? "+" : "-"}${Math.abs(value * 100).toFixed(1)}`;

export const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;
