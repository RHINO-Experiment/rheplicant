from pathlib import Path

CSS = (
    Path(__file__).parents[2] / "src/rheplicant/gui/react/editor.css"
).read_text(encoding="utf-8")


def _luminance(colour: str) -> float:
    channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92
        if value <= 0.04045
        else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_editor_text_and_error_surfaces_have_wcag_aa_contrast():
    pairs = (
        ("#17251d", "#ffffff"),
        ("#7a1010", "#fff1f0"),
        ("#f4f7f3", "#111814"),
        ("#ffd6d2", "#3a1714"),
    )

    assert all(first in CSS and second in CSS for first, second in pairs)
    assert all(_contrast(first, second) >= 4.5 for first, second in pairs)


def test_focus_indicator_is_thick_and_contrasts_in_light_and_dark_modes():
    assert ":focus-visible" in CSS
    assert "outline: 3px solid #005fcc" in CSS
    assert "outline-color: #8db9ff" in CSS
    assert _contrast("#005fcc", "#ffffff") >= 3
    assert _contrast("#8db9ff", "#111814") >= 3
    assert "@media (forced-colors: active)" in CSS
