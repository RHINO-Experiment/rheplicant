from pathlib import Path

CSS = (
    Path(__file__).parents[2] / "src/rheplicant/gui/react/editor.css"
).read_text(encoding="utf-8")
TOKENS = (
    Path(__file__).parents[2] / "src/rheplicant/gui/react/tokens.css"
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
        ("#17212b", "#ffffff"),
        ("#8f1d24", "#fdebec"),
        ("#eef3f7", "#111820"),
        ("#ffb3ba", "#492126"),
    )

    assert all(first in TOKENS and second in TOKENS for first, second in pairs)
    assert all(_contrast(first, second) >= 4.5 for first, second in pairs)


def test_focus_indicator_is_thick_and_contrasts_in_light_and_dark_modes():
    assert ":focus-visible" in CSS
    assert "outline: 3px solid var(--rh-focus)" in CSS
    assert "#005fcc" in TOKENS
    assert "#8dc8ff" in TOKENS
    assert _contrast("#005fcc", "#ffffff") >= 3
    assert _contrast("#8dc8ff", "#111820") >= 3
    assert "@media (forced-colors: active)" in TOKENS


def test_editor_uses_shared_tokens_with_dark_and_forced_colour_overrides():
    assert "var(--rh-text)" in CSS
    assert "var(--rh-danger-text)" in CSS
    assert "@media (prefers-color-scheme: dark)" in TOKENS
    assert "@media (forced-colors: active)" in TOKENS
    assert "--rh-focus: Highlight" in TOKENS


def test_wide_workbench_is_a_bounded_frame_with_independent_main_and_inspector_scroll():
    assert "box-sizing: border-box;" in CSS
    assert "height: 100dvh;" in CSS
    assert "grid-template-rows: minmax(0, 14rem) minmax(0, 1fr) minmax(0, 16rem);" in CSS
    assert "grid-template-columns: minmax(14rem, 16rem) minmax(0, 1fr) minmax(22rem, 28rem);" in CSS
    assert ".workbench-main," in CSS
    assert ".workbench-inspector" in CSS
    assert CSS.count("min-height: 0;") >= 2
    assert CSS.count("min-width: 0;") >= 2
    assert CSS.count("overflow: auto;") >= 2
