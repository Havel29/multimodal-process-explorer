import argparse

import pandas as pd
import pytest

from app.config import load_config
from scripts.prepare_demo_data import choose_cases, concise_summary, render_event, resolve_inputs

CONFIG = load_config()


def arguments(**overrides):
    return argparse.Namespace(**{"log": None, "images": None, "without_images": False, **overrides})


def write_log(tmp_path):
    log = tmp_path / "covid_log.csv"
    log.write_text("patient_id,activity,time:timestamp,outcome,path\n", encoding="utf-8")
    return log


def test_render_event_keeps_the_authors_text_format():
    assert render_event(pd.Series({"activity": "entered in ICU"}), 3600, CONFIG) == "Patient entered in ICU 3600 seconds ago."


def test_unknown_activities_use_the_default_template():
    assert render_event(pd.Series({"activity": "was transferred"}), 0, CONFIG) == "Patient event: was transferred."


def test_summary_is_bounded():
    assert len(concise_summary("x" * 300, 230)) == 230


def test_resolve_inputs_uses_explicit_log_and_images(tmp_path):
    log = write_log(tmp_path)
    images = tmp_path / "images"
    images.mkdir()

    assert resolve_inputs(arguments(log=log, images=images), CONFIG) == (log, images)


def test_resolve_inputs_explains_that_log_does_not_contain_images(tmp_path):
    with pytest.raises(SystemExit, match="CSV contains image paths"):
        resolve_inputs(arguments(log=write_log(tmp_path), images=tmp_path / "missing"), CONFIG)


def test_resolve_inputs_allows_pathway_only_mode(tmp_path):
    log = write_log(tmp_path)

    assert resolve_inputs(arguments(log=log, without_images=True), CONFIG) == (log, None)


def test_choose_cases_balances_outcomes_and_lists_the_risk_class_last():
    rows = []
    for patient_id, outcome in [(1, "Home"), (2, "Death"), (3, "Home"), (4, "Death")]:
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=patient_id)
        rows += [
            {"patient_id": patient_id, "activity": "entered in ED", "time:timestamp": start, "outcome": outcome, "path": None},
            {"patient_id": patient_id, "activity": "performed the digital radiography",
             "time:timestamp": start + pd.Timedelta(hours=1), "outcome": outcome, "path": "['p/image.jpg']"},
        ]
    log = pd.DataFrame(rows)

    selected = choose_cases(log, 2, None, CONFIG)

    outcomes = [group["outcome"].iloc[0] for _, group in selected]
    assert outcomes == ["Home", "Death"]
