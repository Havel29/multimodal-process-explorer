"""Turn token-level attributions into per-event and per-word attributions."""
from __future__ import annotations


def event_attributions(
    history: str,
    spans: list[tuple[str, int, int]],
    tokens: list[tuple[int, int, float]],
    top_words: int = 3,
) -> dict[str, dict]:
    """Sum token scores per event; events outside the token window are left out.

    ``segments`` cover the event's whole text: words with their score, and the text between them with none.
    """
    per_event: dict[str, dict] = {}
    bounds = {event_id: (start, end) for event_id, start, end in spans}
    span_index = 0
    for start, end, score in sorted(tokens):
        while span_index < len(spans) and spans[span_index][2] <= start:
            span_index += 1
        if span_index == len(spans):
            break
        event_id, span_start, _ = spans[span_index]
        if start < span_start:
            continue
        entry = per_event.setdefault(event_id, {"score": 0.0, "words": []})
        entry["score"] += score
        _append_piece(entry["words"], history[start:end], start, end, score)

    return {
        event_id: {
            "score": round(entry["score"], 6),
            "tokens": _strongest(entry["words"], top_words),
            "segments": _segments(history, *bounds[event_id], entry["words"]),
        }
        for event_id, entry in per_event.items()
    }


def history_windows(
    spans: list[tuple[str, int, int]], token_counts: list[int], limit: int
) -> list[list[tuple[str, int, int]]]:
    """Split events into consecutive windows of at most ``limit`` tokens, filled from the latest event.

    An event longer than ``limit`` gets a window of its own (and is truncated by the model).
    """
    windows, current, size = [], [], 0
    for span, count in reversed(list(zip(spans, token_counts))):
        if current and size + count > limit:
            windows.append(current[::-1])
            current, size = [], 0
        current.append(span)
        size += count
    if current:
        windows.append(current[::-1])
    return windows[::-1]


def _append_piece(words: list[list], piece: str, start: int, end: int, score: float) -> None:
    """Join WordPiece fragments back into whole words: [text, start, end, score]."""
    previous = words[-1] if words else None
    if previous and previous[2] == start and piece.isalnum() and previous[0][-1:].isalnum():
        previous[0] += piece
        previous[2] = end
        previous[3] += score
    else:
        words.append([piece, start, end, score])


def _segments(history: str, start: int, end: int, words: list[list]) -> list[list]:
    """The event text from ``start`` to ``end`` as [text, score or None] pieces, in order."""
    segments, cursor = [], start
    for _, word_start, word_end, score in words:
        if word_start > cursor:
            segments.append([history[cursor:word_start], None])
        segments.append([history[word_start:word_end], round(score, 6)])
        cursor = word_end
    if cursor < end:
        segments.append([history[cursor:end], None])
    return segments


def _strongest(words: list[list], count: int) -> list[dict]:
    totals: dict[str, float] = {}
    for word, _, _, score in words:
        if any(char.isalnum() for char in word):
            totals[word] = totals.get(word, 0.0) + score
    ranked = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)[:count]
    return [{"text": word, "score": round(score, 6)} for word, score in ranked]
