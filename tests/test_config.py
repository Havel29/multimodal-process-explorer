import pytest

from app.config import OTHER_EVENT_TYPE, load_config

from .conftest import ROOT


def test_cdsl_configuration_keeps_the_authors_class_order():
    config = load_config()

    assert [outcome.source_value for outcome in config.outcome_classes] == ["Death", "Home"]
    assert config.class_index(config.risk_class) == 0


def test_activities_map_to_event_types_and_stages():
    config = load_config()

    assert config.event_type_of("received vital signs check").key == "vitals"
    assert config.event_type_of("unknown activity") is OTHER_EVENT_TYPE
    assert config.is_prediction_stage("performed the digital radiography")
    assert not config.is_prediction_stage("performed the computed tomography")


def test_public_configuration_exposes_no_file_paths():
    public = load_config().public()

    assert "checkpoint" not in str(public)
    assert "maxCutoffDay" not in public
    assert {model["id"] for model in public["models"]} == {"roberta", "flava", "vit"}


def test_unknown_model_kind_is_rejected(tmp_path):
    text = (ROOT / "config" / "cdsl_covid.toml").read_text(encoding="utf-8").replace('kind = "image"', 'kind = "audio"')
    path = tmp_path / "broken.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="unknown kind"):
        load_config(path)


def test_texts_keep_their_defaults_fill_in_the_nouns_and_take_overrides():
    from app.config import config_from_mapping

    config = config_from_mapping({"demo": {"stage_noun": "scan", "pathway_noun": "claim"},
                                  "texts": {"why_title": "What drives it?"}})
    texts = config.resolved_texts()

    assert texts["why_title"] == "What drives it?"
    assert "any scan marker" in texts["timeline_help"]
    assert "along each claim:" in texts["statistics_intro"]
    assert "{last_day}" in texts["statistics_caveat"]  # filled in by the page


def test_unknown_text_keys_are_rejected():
    import pytest

    from app.config import config_from_mapping

    with pytest.raises(ValueError, match="why_titel"):
        config_from_mapping({"texts": {"why_titel": "typo"}})


def test_saved_configuration_lists_every_text():
    import tomllib

    from app.config import config_from_mapping
    from app.config_writer import dump_config
    from app.texts import DEFAULT_TEXTS

    config = config_from_mapping({"texts": {"why_title": "What drives it?"},
                                  "explanation": {"model": "multimodal", "method": "explainers/example_occlusion.py",
                                                  "options": {"max_sentences": 10}}})
    saved = tomllib.loads(dump_config(config))

    assert set(saved["texts"]) == set(DEFAULT_TEXTS)
    assert saved["texts"]["why_title"] == "What drives it?"
    assert saved["explanation"][0]["options"] == {"max_sentences": 10}
    assert config_from_mapping(saved).resolved_texts() == config.resolved_texts()


def test_page_fields_do_not_clash_with_noun_plurals():
    import re

    from app.texts import DEFAULT_TEXTS, NOUNS

    nouns = set(NOUNS) | {f"{noun}s" for noun in NOUNS}
    page_fields = {"event_count", "day_count", "image_count", "model", "method", "risk", "count", "last_day",
                   "minimum", "table_day"}
    used = {field for text in DEFAULT_TEXTS.values() for field in re.findall(r"\{(\w+)\}", text)}
    assert used <= nouns | page_fields
    assert not page_fields & nouns


def test_a_configuration_takes_one_model_per_role_at_most():
    from app.config import config_from_mapping

    single = config_from_mapping({"models": [{"kind": "image", "checkpoint": "x.pth"}]})
    # The built-in Integrated Gradients only explains the multimodal model, so no explanation by default
    assert [model.kind for model in single.models] == ["image"]
    assert single.explanations == ()
    with pytest.raises(ValueError, match="one model per role"):
        config_from_mapping({"models": [{"kind": "text", "id": "a"}, {"kind": "text", "id": "b"}]})
