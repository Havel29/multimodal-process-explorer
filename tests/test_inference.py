import tomllib

import pandas as pd
import pytest

from app.config import config_from_mapping, load_config
from app.config_writer import dump_config
from app.inference import complete_config
from scripts.prepare_demo_data import render_event

COLUMNS = {"image": "photo", "outcome": "case:outcome"}


def claims_log():
    rows = []
    for case, outcome in [("c1", "approved"), ("c2", "approved"), ("c3", "rejected")]:
        rows += [
            {"case:concept:name": case, "concept:name": "Submit claim", "amount": 6461.0, "photo": None},
            {"case:concept:name": case, "concept:name": "Upload photo", "amount": None, "photo": f"{case}/1.jpg"},
            {"case:concept:name": case, "concept:name": "Close claim", "amount": None, "photo": None},
        ]
        for row in rows[-3:]:
            row["case:outcome"] = outcome
    log = pd.DataFrame(rows)
    log["time:timestamp"] = pd.Timestamp("2026-01-01")
    log["@@index"] = range(len(log))
    return log


def generic_config(**sections):
    return config_from_mapping({"preparation": {"columns": COLUMNS}, **sections})


def test_an_empty_configuration_loads_with_defaults():
    config = config_from_mapping({})

    assert config.event_types == () and config.outcome_classes == ()
    assert {model.kind for model in config.models} == {"text", "image", "flava"}
    assert config.preparation.columns["case"] == "case:concept:name"


def test_lanes_image_events_and_classes_are_read_from_the_log():
    config, notes = complete_config(generic_config(), claims_log())

    assert [event_type.label for event_type in config.event_types] == ["Submit claim", "Upload photo", "Close claim"]
    assert config.is_prediction_stage("Upload photo") and not config.is_prediction_stage("Submit claim")
    assert [outcome.source_value for outcome in config.outcome_classes] == ["approved", "rejected"]
    assert config.risk_class == "rejected"  # the rarer outcome
    assert any("a guess" in note for note in notes)


def test_what_the_configuration_says_is_kept():
    given = generic_config(
        outcome={"risk_class": "ok", "classes": [
            {"code": "no", "source_value": "rejected", "label": "No"},
            {"code": "ok", "source_value": "approved", "label": "Yes"},
        ]},
        event_types=[{"key": "photos", "label": "Photos", "color": "#000000",
                      "activities": [{"name": "Upload photo", "prediction_stage": True}]}],
    )
    config, notes = complete_config(given, claims_log())

    assert [outcome.code for outcome in config.outcome_classes] == ["no", "ok"] and config.risk_class == "ok"
    assert [event_type.key for event_type in config.event_types] == ["photos"]
    assert any("shown as Other" in note for note in notes)


def test_a_label_value_outside_the_configured_classes_is_an_error():
    given = generic_config(outcome={"risk_class": "no", "classes": [{"code": "no", "source_value": "rejected", "label": "No"}]})

    with pytest.raises(ValueError, match="approved"):
        complete_config(given, claims_log())


def test_without_classes_or_a_label_the_setup_cannot_go_on():
    with pytest.raises(ValueError, match="label field"):
        complete_config(config_from_mapping({"preparation": {"columns": {"image": "photo"}}}), claims_log())


def test_generic_texts_use_the_attributes_but_never_the_label():
    config, _ = complete_config(generic_config(), claims_log())
    row = claims_log().iloc[0]

    assert render_event(row, 0.0, config) == "Submit claim; amount: 6461."


def test_a_completed_configuration_survives_a_round_trip():
    config, _ = complete_config(generic_config(), claims_log())
    reloaded = config_from_mapping(tomllib.loads(dump_config(config, header="test")))

    assert reloaded.event_types == config.event_types
    assert reloaded.outcome_classes == config.outcome_classes
    assert reloaded.risk_class == config.risk_class
    assert reloaded.preparation.columns == config.preparation.columns


def test_the_cdsl_configuration_survives_a_round_trip():
    config = load_config()
    reloaded = config_from_mapping(tomllib.loads(dump_config(config)))

    assert reloaded.event_types == config.event_types
    assert reloaded.outcome_classes == config.outcome_classes
    assert reloaded.preparation == config.preparation
