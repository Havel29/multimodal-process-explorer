"""A synthetic dataset with many event types, to test the interface at scale."""
from __future__ import annotations

import base64
import colorsys
import json
import random
from pathlib import Path

from app.config import PROJECT_ROOT

SPECIALTIES = ["haematology", "chemistry", "microbiology", "cardiology", "respiratory", "nursing", "pharmacy", "nutrition"]


def _color(index: int, count: int) -> str:
    red, green, blue = colorsys.hls_to_rgb(index / count, 0.45, 0.55)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def write_config(directory: Path, types: int) -> Path:
    """A copy of the CDSL configuration with ``types`` event types plus one image type."""
    base = (PROJECT_ROOT / "config" / "cdsl_covid.toml").read_text(encoding="utf-8")
    head = base[: base.index("[[event_types]]")]
    tail = base[base.index("[preparation]"):]
    head = head.replace('bundle = "data/curated/demo.json"', f'bundle = "{(directory / "demo.json").as_posix()}"')
    head = head.replace('analytics_dir = "data/curated"', f'analytics_dir = "{directory.as_posix()}"')
    lanes = []
    for index in range(types):
        lanes.append(
            f'[[event_types]]\nkey = "type{index:02d}"\nlabel = "{SPECIALTIES[index % len(SPECIALTIES)].capitalize()} {index:02d}"\n'
            f'color = "{_color(index, types)}"\n\n  [[event_types.activities]]\n  name = "activity {index:02d}"\n'
            f'  display = "Measurement {index:02d} ({SPECIALTIES[index % len(SPECIALTIES)]})"\n  template = "Patient had measurement {index:02d}."\n'
        )
    lanes.append(
        '[[event_types]]\nkey = "xray"\nlabel = "X-ray"\ncolor = "#14283d"\n\n  [[event_types.activities]]\n'
        '  name = "performed the digital radiography"\n  display = "Digital radiography"\n  short = "DX"\n'
        '  prediction_stage = true\n  template = "Patient performed a radiography."\n'
    )
    path = directory / f"synthetic_{types}.toml"
    path.write_text(head + "\n".join(lanes) + "\n" + tail, encoding="utf-8")
    return path


def write_bundle(directory: Path, types: int, patients: int = 3, events: int = 400, seed: int = 7) -> Path:
    """Patients with ``events`` events of random types over two weeks, and three radiographs each."""
    rng = random.Random(seed)
    (directory / "images").mkdir(parents=True, exist_ok=True)
    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZQmcAAAAASUVORK5CYII="
    )
    bundle = {"metadata": {"source": "synthetic", "mode": "multimodal"}, "patients": []}
    for number in range(1, patients + 1):
        offsets = sorted(rng.randrange(0, 14 * 86400) for _ in range(events))
        items = []
        for index, offset in enumerate(offsets, start=1):
            kind = rng.randrange(types)
            text = f" Patient had measurement {kind:02d} {offset} seconds ago."
            items.append({"id": f"E{index:03d}", "activity": f"activity {kind:02d}", "offset_seconds": offset,
                          "summary": text.strip(), "model_text": text, "prediction_stage": False})
        for day in (2, 6, 10):
            target = directory / "images" / f"P{number}_{day}.png"
            target.write_bytes(image)
            items.append({"id": f"X{day:02d}", "activity": "performed the digital radiography", "offset_seconds": day * 86400 + 3600,
                          "summary": "Patient performed a radiography.", "model_text": " Patient performed a radiography.",
                          "prediction_stage": True, "image": f"images/{target.name}"})
        items.sort(key=lambda item: item["offset_seconds"])
        bundle["patients"].append({"id": f"Patient {number:02d}", "true_outcome": "Death" if number % 2 else "Home",
                                   "duration_seconds": items[-1]["offset_seconds"], "events": items})
    path = directory / "demo.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path
