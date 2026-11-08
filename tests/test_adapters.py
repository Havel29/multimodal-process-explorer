"""Models given as adapter files ([[models]] adapter = "...")."""
import pytest

from app.config import config_from_mapping
from app.config_writer import dump_config
from app.models import adapter_class, build_models

EXAMPLE = "adapters/example_bag_of_words.py"
OUTCOMES = {"risk_class": "bad", "classes": [
    {"code": "bad", "source_value": "bad", "label": "Bad"}, {"code": "good", "source_value": "good", "label": "Good"}]}


def config_with(checkpoint, adapter=EXAMPLE):
    return config_from_mapping({"outcome": OUTCOMES, "models": [
        {"kind": "text", "id": "words", "name": "Bag of words", "adapter": adapter, "checkpoint": str(checkpoint)},
        {"kind": "image", "checkpoint": str(checkpoint)},
        {"kind": "flava", "checkpoint": str(checkpoint)},
    ]})


def test_an_adapter_file_builds_loads_its_checkpoint_and_predicts(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "words.pth"
    config = config_with(checkpoint)
    model = build_models(config, "cpu")["words"]
    assert type(model).__name__ == "Model"

    torch.save(model._network().state_dict(), checkpoint)  # the checkpoint format the example documents
    probabilities = model.predict("Emergency admission; triage: urgent.", None)

    assert len(probabilities) == 2 and abs(sum(probabilities) - 1) < 1e-5


def test_a_safetensors_checkpoint_loads_too(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    checkpoint = tmp_path / "words.safetensors"
    model = build_models(config_with(checkpoint), "cpu")["words"]
    save_file(model._network().state_dict(), str(checkpoint))

    assert len(model.predict("Vital signs; oxygen_saturation: 88.", None)) == 2


def test_a_missing_adapter_file_is_reported(tmp_path):
    with pytest.raises(ValueError, match="adapter file 'adapters/missing.py' not found"):
        config_with(tmp_path / "x.pth", adapter="adapters/missing.py")


def test_an_adapter_file_without_a_model_class_is_rejected(tmp_path):
    empty = tmp_path / "empty.py"
    empty.write_text("x = 1\n", encoding="utf-8")
    config = config_with(tmp_path / "x.pth", adapter=str(empty))

    with pytest.raises(ValueError, match="no class Model"):
        adapter_class(config.models[0])


def test_the_saved_configuration_keeps_the_adapter(tmp_path):
    saved = dump_config(config_with(tmp_path / "x.pth"))

    assert f'adapter = "{EXAMPLE}"' in saved
    assert saved.count("adapter =") == 1  # built-in models write no adapter key
