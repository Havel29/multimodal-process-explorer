#!/usr/bin/env python3
"""Generate the open example: synthetic pediatric respiratory admissions with real chest X-rays.

The radiographs come from the Kermany et al. chest X-ray dataset (children aged one to five, CC BY 4.0,
https://doi.org/10.17632/rscbjbr9sj.2); the pathways around them are synthetic. A child is admitted
with respiratory symptoms, has laboratory tests, one or two chest X-rays, vital signs and maybe
antibiotics and oxygen, and is either discharged home or transferred to intensive care. The outcome
needs both modalities: the transfer follows a pneumonia on a radiograph (which only the image shows)
together with an urgent triage at admission or a respiratory deterioration during the stay (which only
the history says). Information arrives over the stay: a deterioration is recorded on the day it
happens, and some pneumonias show only on the follow-up X-ray. Nothing in the pathway depends on the
outcome.

Writes the training log (data/xray-pathways/pathways_full.xes, from the dataset's training images),
a test log (data/xray-pathways/pathways_test.xes, from its test images) and the 20 cases the explorer
loads (examples/xray-pathways/pathways.xes, from its test images, 10 per outcome), each with its
radiographs.

    python scripts/make_xray_pathways.py --zip data/xray-pathways/source/ChestXRay2017.zip
"""
from __future__ import annotations

import argparse
import io
import random
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parents[1]
SYMPTOMS = ["cough and fever", "fever and fast breathing", "cough and difficulty breathing", "wheezing and cough",
            "fever, cough and poor feeding"]
ANTIBIOTICS = ["amoxicillin", "ampicillin", "ceftriaxone"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=ROOT / "data" / "xray-pathways" / "source" / "ChestXRay2017.zip",
                        help="ChestXRay2017.zip from https://doi.org/10.17632/rscbjbr9sj.2")
    parser.add_argument("--cases", type=int, default=2000, help="Training cases")
    parser.add_argument("--test-cases", type=int, default=500, help="Test cases (scripts/evaluate_models.py)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data" / "xray-pathways")
    parser.add_argument("--example-dir", type=Path, default=ROOT / "examples" / "xray-pathways")
    parser.add_argument("--image-size", type=int, default=512, help="Longer side of the stored radiographs")
    return parser.parse_args()


def radiographs(archive: zipfile.ZipFile) -> dict[str, dict[bool, list[str]]]:
    """Image names by split ("train", "test") and finding (True: pneumonia)."""
    found: dict[str, dict[bool, list[str]]] = {"train": {True: [], False: []}, "test": {True: [], False: []}}
    for name in archive.namelist():
        parts = name.split("/")
        if "__MACOSX" in parts or not name.lower().endswith((".jpeg", ".jpg")):
            continue
        split = "train" if "train" in parts else "test" if "test" in parts else None
        if split:
            found[split]["PNEUMONIA" in parts].append(name)
    return found


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_case(rng: random.Random, number: int, start: datetime) -> tuple[list[dict], str, list[bool]]:
    """Events of one admission (radiographs still to attach), its outcome and the finding of each X-ray."""
    # The outcome follows what happens, never the other way round. Urgency at admission, a deterioration
    # during the stay and the finding (radiographs only) are independent; the transfer needs
    # a pneumonia seen on a radiograph together with urgency or deterioration. Information arrives over
    # the stay: a deterioration shows from its day on, and a pneumonia may show only on the follow-up X-ray.
    pneumonia = rng.random() < 0.4
    spo2 = round(clip(rng.gauss(94, 3), 82, 100))
    crp = round(clip(rng.gauss(50, 35), 1, 250))
    temperature = round(clip(rng.gauss(38.3, 0.6), 36.5, 40.5), 1)
    case = f"P{number:05d}"
    t = start
    urgent = spo2 < 93
    events = [{"activity": "Emergency admission", "time": t, "age_years": rng.randint(1, 5),
               "sex": rng.choice(["female", "male"]), "symptoms": rng.choice(SYMPTOMS),
               "triage": "urgent" if urgent else "standard",
               "temperature": temperature, "oxygen_saturation": spo2}]
    events.append({"activity": "Laboratory tests", "time": t + timedelta(hours=1),
                   "crp_mg_l": crp, "white_cells_10e9_l": round(clip(rng.gauss(11.5, 4), 3, 35), 1)})
    xray_time = t + timedelta(hours=rng.randint(2, 6))  # after the tests, so that it knows them
    events.append({"activity": "Chest X-ray", "time": xray_time, "view": "anteroposterior"})
    findings = [pneumonia]

    if spo2 < 92:
        events.append({"activity": "Oxygen therapy started", "time": t + timedelta(hours=rng.randint(1, 4)),
                       "flow_l_min": rng.choice([1, 2, 3])})
    antibiotics_hours = None
    if rng.random() < 0.55:
        antibiotics_hours = rng.choice([rng.randint(2, 20), rng.randint(2, 20), rng.randint(30, 60)])
        events.append({"activity": "Antibiotics started", "time": t + timedelta(hours=antibiotics_hours),
                       "drug": rng.choice(ANTIBIOTICS), "route": rng.choice(["oral", "intravenous"])})

    days = rng.randint(3, 6)
    deterioration_day = rng.randint(2, days - 1) if rng.random() < 0.3 else None
    if deterioration_day is not None:
        events.append({"activity": "Respiratory deterioration",
                       "time": t + timedelta(days=deterioration_day, hours=rng.randint(2, 5)),
                       "oxygen_saturation": round(clip(spo2 - 7 + rng.gauss(0, 1.5), 75, 95)),
                       "respiratory_rate": round(clip(rng.gauss(52, 5), 30, 80))})
    for day in range(1, days + 1):
        worse = deterioration_day is not None and day >= deterioration_day
        events.append({"activity": "Vital signs", "time": t + timedelta(days=day, hours=rng.randint(6, 10)),
                       "temperature": round(clip(temperature - 0.3 * day + (0.8 if worse else 0) + rng.gauss(0, 0.3), 36.2, 40.5), 1),
                       "oxygen_saturation": round(clip((spo2 - 6 if worse else spo2 + 1.0 * day) + rng.gauss(0, 1.5), 75, 100)),
                       "respiratory_rate": round(clip(rng.gauss(46 if worse else 36 if spo2 < 93 else 30, 5), 18, 80))})
    if rng.random() < 0.5:
        events.append({"activity": "Follow-up chest X-ray", "time": t + timedelta(days=rng.randint(2, days), hours=rng.randint(11, 16)),
                       "view": "anteroposterior"})
        # A pneumonia not visible at admission may develop and show only now
        pneumonia = pneumonia or rng.random() < 0.3
        findings.append(pneumonia)

    intensive = pneumonia and (urgent or deterioration_day is not None)
    if rng.random() < 0.05:  # a few courses do not follow the rule
        intensive = not intensive
    events.sort(key=lambda event: event["time"])
    for event in events:
        event["case"] = case
    return events, ("intensive care" if intensive else "home"), findings


def attach_radiographs(events: list[dict], findings: list[bool], rng: random.Random, archive: zipfile.ZipFile,
                       pool: dict[bool, list[str]], folder: Path, size: int) -> None:
    from PIL import Image

    xrays = [event for event in events if event["activity"].endswith("chest X-ray") or event["activity"] == "Chest X-ray"]
    xrays.sort(key=lambda event: event["activity"] != "Chest X-ray")  # the admission X-ray first
    for number, (event, pneumonia) in enumerate(zip(xrays, findings), start=1):
        source = rng.choice(pool[pneumonia])
        image = Image.open(io.BytesIO(archive.read(source))).convert("L")
        image.thumbnail((size, size))
        relative = Path(event["case"]) / f"xray{number}.jpg"
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        image.save(folder / relative, quality=88)
        event["image"] = relative.as_posix()


def write_xes(cases: list[tuple[list[dict], str]], path: Path) -> None:
    """A plain XES file with only the attributes each event has."""
    def attribute(key: str, value) -> str:
        if isinstance(value, datetime):
            return f'<date key={quoteattr(key)} value="{value.strftime("%Y-%m-%dT%H:%M:%S")}+00:00"/>'
        kind = "int" if isinstance(value, int) else "float" if isinstance(value, float) else "string"
        return f"<{kind} key={quoteattr(key)} value={quoteattr(str(value))}/>"

    names = {"activity": "concept:name", "time": "time:timestamp"}
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<log xes.version="1.0" xmlns="http://www.xes-standard.org/">',
             '  <extension name="Concept" prefix="concept" uri="http://www.xes-standard.org/concept.xesext"/>',
             '  <extension name="Time" prefix="time" uri="http://www.xes-standard.org/time.xesext"/>']
    for events, outcome in cases:
        lines += ["  <trace>", f"    {attribute('concept:name', events[0]['case'])}"]
        for event in events:
            lines.append("    <event>")
            lines += [f"      {attribute(names.get(key, key), value)}" for key, value in event.items() if key != "case"]
            lines += [f"      {attribute('outcome', outcome)}", "    </event>"]
        lines.append("  </trace>")
    lines.append("</log>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    archive = zipfile.ZipFile(args.zip)
    pools = radiographs(archive)
    start = datetime(2025, 1, 6, 8)

    train = []
    for number in range(1, args.cases + 1):
        events, outcome, findings = make_case(rng, number, start + timedelta(hours=number * 4))
        attach_radiographs(events, findings, rng, archive, pools["train"], args.train_dir / "images", args.image_size)
        train.append((events, outcome))
    write_xes(train, args.train_dir / "pathways_full.xes")
    counts = {outcome: sum(case[1] == outcome for case in train) for outcome in ("home", "intensive care")}
    print(f"{len(train)} training cases from the dataset's training images: {counts}")

    # Test cases, from the dataset's test images, for scripts/evaluate_models.py only: neither training
    # nor the choice of the checkpoints sees them
    test = []
    for number in range(args.cases + 1, args.cases + args.test_cases + 1):
        events, outcome, findings = make_case(rng, number, start + timedelta(hours=number * 4))
        attach_radiographs(events, findings, rng, archive, pools["test"], args.train_dir / "images", args.image_size)
        test.append((events, outcome))
    write_xes(test, args.train_dir / "pathways_test.xes")
    counts = {outcome: sum(case[1] == outcome for case in test) for outcome in ("home", "intensive care")}
    print(f"{len(test)} test cases from the dataset's test images: {counts}")

    # The cases shown in the explorer use the dataset's test images, 10 per outcome
    held = {"home": [], "intensive care": []}
    number = args.cases + args.test_cases
    while min(map(len, held.values())) < 10:
        number += 1
        events, outcome, findings = make_case(rng, number, start + timedelta(hours=number * 4))
        if len(held[outcome]) < 10:
            attach_radiographs(events, findings, rng, archive, pools["test"], args.example_dir / "images", args.image_size)
            held[outcome].append((events, outcome))
    example = sorted(held["home"] + held["intensive care"], key=lambda case: case[0][0]["case"])
    write_xes(example, args.example_dir / "pathways.xes")
    print(f"20 example cases in {args.example_dir / 'pathways.xes'}")


if __name__ == "__main__":
    main()
