import pytest

from app.attribution import event_attributions, history_windows
from app.cache import AnalyticsCache
from app.cutoffs import CutoffError, compose_history, resolve_cutoff, trajectory_points

from .conftest import EVENTS


def test_free_cutoff_includes_events_up_to_the_time_point():
    known = resolve_cutoff(EVENTS, cutoff_seconds=100000)

    assert [event["id"] for event in known.available] == ["E001", "E002", "E003"]
    assert known.image_event["id"] == "E003"


def test_only_one_kind_of_cutoff_is_accepted():
    with pytest.raises(CutoffError):
        resolve_cutoff(EVENTS, cutoff_day=1, cutoff_seconds=100)


def test_day_cutoff_uses_the_case_duration_as_its_horizon():
    known = resolve_cutoff(EVENTS, cutoff_day=3)

    assert known.available == EVENTS
    with pytest.raises(CutoffError):
        resolve_cutoff(EVENTS, cutoff_day=4)


def test_history_spans_point_at_each_enabled_event():
    history, spans = compose_history(EVENTS, {"E002", "E004"})

    assert history == "day one fever CEFTRIAXONA "
    assert [(event_id, history[start:end]) for event_id, start, end in spans] == [
        ("E002", "day one fever "), ("E004", "CEFTRIAXONA "),
    ]


def test_trajectory_points_cover_every_day_of_the_stay_and_each_radiograph():
    points = trajectory_points({"events": EVENTS})

    assert [(point["kind"], point.get("day") or point.get("stage_event_id")) for point in points] == [
        ("day", 1), ("stage", "E003"), ("day", 2), ("day", 3),
    ]


def test_token_attributions_are_summed_per_event_and_wordpieces_rejoined():
    history, spans = compose_history(EVENTS, {"E002", "E004"})
    tokens = [(8, 13, 0.1), (14, 20, 0.2), (20, 22, 0.3), (22, 25, -0.05)]  # fever, CEFTRI, AX, ONA

    events = event_attributions(history, spans, tokens)

    assert events["E002"]["score"] == pytest.approx(0.1)
    assert events["E004"]["score"] == pytest.approx(0.45)
    assert events["E004"]["tokens"] == [{"text": "CEFTRIAXONA", "score": pytest.approx(0.45)}]


def test_events_outside_the_token_window_get_no_attribution():
    history, spans = compose_history(EVENTS, {"E001", "E004"})

    events = event_attributions(history, spans, [(10, 21, 0.4)])

    assert set(events) == {"E004"}


def test_history_windows_respect_the_limit_and_start_from_the_latest_event():
    spans = [("E1", 0, 1), ("E2", 1, 2), ("E3", 2, 3), ("E4", 3, 4)]

    windows = history_windows(spans, [3, 2, 2, 1], limit=4)

    assert [[event_id for event_id, _, _ in window] for window in windows] == [["E1"], ["E2"], ["E3", "E4"]]


def test_an_event_longer_than_the_limit_gets_its_own_window():
    windows = history_windows([("E1", 0, 1), ("E2", 1, 2)], [9, 1], limit=4)

    assert [[event_id for event_id, _, _ in window] for window in windows] == [["E1"], ["E2"]]


def test_a_result_computed_across_a_reset_is_not_saved(tmp_path):
    bundle = tmp_path / "demo.json"
    bundle.write_text("{}", encoding="utf-8")
    cache = AnalyticsCache(tmp_path / "analytics.json", bundle)

    def compute():
        cache.clear()  # the reset arrives while the case is being computed
        return [1]

    assert cache.get_or_compute("Patient 01", compute) == [1]
    assert cache.get("Patient 01") is None


def test_analytics_cache_is_invalidated_when_the_bundle_changes(tmp_path):
    bundle = tmp_path / "demo.json"
    bundle.write_text("{}", encoding="utf-8")
    cache = AnalyticsCache(tmp_path / "analytics.json", bundle)
    calls = []

    cache.get_or_compute("Patient 01", lambda: calls.append(1) or [1])
    cache.get_or_compute("Patient 01", lambda: calls.append(1) or [1])
    assert len(calls) == 1

    bundle.write_text('{"changed": true}', encoding="utf-8")
    assert cache.get("Patient 01") is None


def test_segments_give_every_word_of_the_event_its_score_and_keep_the_text_between():
    history, spans = compose_history(EVENTS, {"E002", "E004"})
    tokens = [(8, 13, 0.1), (14, 20, 0.2), (20, 22, 0.3), (22, 25, -0.05)]  # fever, CEFTRI, AX, ONA

    events = event_attributions(history, spans, tokens)

    assert events["E002"]["segments"] == [["day one ", None], ["fever", pytest.approx(0.1)], [" ", None]]
    assert "".join(text for text, _ in events["E004"]["segments"]) == "CEFTRIAXONA "
    assert events["E004"]["segments"][0] == ["CEFTRIAXONA", pytest.approx(0.45)]
