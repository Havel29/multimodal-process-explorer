import re

from .conftest import ROOT

STATIC = ROOT / "app" / "static"
INTERFACE_FILES = [*STATIC.rglob("*.js"), *STATIC.rglob("*.css"), *(ROOT / "app" / "templates").glob("*.html")]


def test_interface_has_no_motion_effects():
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")

    assert "@keyframes" not in css
    assert "transition:" not in css.replace("transition: none !important", "")


def test_only_the_help_tour_moves_and_not_with_reduced_motion():
    # The tour is the one place with motion, on purpose; it stops when the system asks for less motion.
    tour = re.sub(r"/\*.*?\*/", "", (STATIC / "css" / "tour.css").read_text(encoding="utf-8"), flags=re.DOTALL)
    # The selector of every innermost rule (inside @media and @keyframes too)
    selectors = [part.strip() for rule in re.findall(r"([^{}]+)\{[^{}]*\}", tour) for part in rule.split(",")]

    assert selectors
    keyframe = re.compile(r"^(from|to|\d+%)$")
    outside = [selector for selector in selectors if not selector.startswith(".tour") and not keyframe.match(selector)]
    assert not outside, outside
    assert "@media (prefers-reduced-motion: reduce)" in tour


def test_interface_is_plain_ascii():
    for path in INTERFACE_FILES:
        text = path.read_text(encoding="utf-8")
        assert text.isascii(), path.name
        assert not re.search(r"\\u[0-9a-fA-F]{4}|&(times|hellip|rarr|mdash|ndash|middot|minus);", text), path.name


def test_application_opens_on_the_inputs_page():
    main = (STATIC / "js" / "main.js").read_text(encoding="utf-8")

    # The explorer opens only on #explorer or #statistics with a completed setup
    assert "setup.init({ open: !exploring, canContinue: status.configured," in main
    assert "const exploring = status.configured && [setup.EXPLORER_HASH, setup.STATISTICS_HASH].includes(page);" in main


def test_interface_has_no_dataset_specific_names():
    # Models, event types and outcomes come from the configuration.
    for path in STATIC.rglob("*.js"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"roberta|flava|\bvit\b|non_survival|radiography", text, re.IGNORECASE), path.name


def test_every_module_import_resolves():
    for path in STATIC.rglob("*.js"):
        for target in re.findall(r'from "(\.[^"]+)"', path.read_text(encoding="utf-8")):
            assert (path.parent / target).resolve().is_file(), f"{path.name} imports {target}"
