from pathlib import Path

from rheplicant.core.render import _SVG_STYLE, _THEMES

CSS = (
    Path(__file__).parents[2] / "src/rheplicant/gui/react/editor.css"
).read_text(encoding="utf-8")
TOKENS = (
    Path(__file__).parents[2] / "src/rheplicant/gui/react/tokens.css"
).read_text(encoding="utf-8")


def _block(source: str, marker: str) -> str:
    marker_start = source.index(marker)
    block_start = source.index("{", marker_start)
    depth = 0
    for index in range(block_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[block_start + 1 : index]
    raise AssertionError(f"Unclosed CSS block after {marker}")


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


def test_editor_defines_every_semantic_state_token_in_each_colour_mode():
    semantic_tokens = (
        "--rh-success-bg",
        "--rh-success-text",
        "--rh-warning-bg",
        "--rh-warning-text",
        "--rh-danger-bg",
        "--rh-danger-text",
        "--rh-stale-bg",
        "--rh-stale-text",
        "--rh-disabled-bg",
        "--rh-disabled-text",
    )
    light = _block(TOKENS, ":root")
    dark = _block(TOKENS, "@media (prefers-color-scheme: dark)")
    forced = _block(TOKENS, "@media (forced-colors: active)")

    for token in semantic_tokens:
        assert token in light
        assert token in dark
        assert token in forced


def test_status_chips_use_token_colours_and_a_non_colour_boundary():
    chip = _block(CSS, ".rheplicant-editor .status-chip")

    assert "display: inline-flex;" in chip
    assert "border: 2px solid currentColor;" in chip
    assert "border-radius: var(--rh-radius-sm);" in chip
    for tone, background, text in (
        ("neutral", "--rh-surface-raised", "--rh-text"),
        ("success", "--rh-success-bg", "--rh-success-text"),
        ("warning", "--rh-warning-bg", "--rh-warning-text"),
        ("danger", "--rh-danger-bg", "--rh-danger-text"),
        ("stale", "--rh-stale-bg", "--rh-stale-text"),
        ("disabled", "--rh-disabled-bg", "--rh-disabled-text"),
    ):
        rule = _block(CSS, f".rheplicant-editor .status-{tone}")
        assert f"var({background})" in rule
        assert f"var({text})" in rule


def test_wide_workbench_is_a_bounded_shrink_safe_desktop_frame():
    grid_children = _block(CSS, ".workbench-shell > *")
    assert "box-sizing: border-box;" in CSS
    assert "height: 100dvh;" in CSS
    assert "grid-template-rows: minmax(0, 14rem) minmax(0, 1fr) minmax(0, 16rem);" in CSS
    assert "grid-template-columns: minmax(14rem, 16rem) minmax(0, 1fr) minmax(22rem, 28rem);" in CSS
    assert ".workbench-shell > *," in CSS
    assert ".workbench-layout > *" in CSS
    assert ".product-grid > *," in CSS
    assert ".result-grid > *" in CSS
    assert "min-width: 0;" in grid_children
    assert ".workbench-main," in CSS
    assert ".workbench-inspector" in CSS
    assert CSS.count("min-height: 0;") >= 2
    assert CSS.count("min-width: 0;") >= 2
    assert CSS.count("overflow-y: auto;") >= 2


def test_inspector_uses_border_box_at_full_width_and_height():
    workbench_regions = _block(CSS, ".workbench-header,")

    assert ".workbench-inspector" in CSS
    assert "box-sizing: border-box;" in workbench_regions


def test_workbench_has_all_four_responsive_layout_contracts():
    inspector_overlay = _block(CSS, "@media (max-width: 1279px)")
    compact = _block(CSS, "@media (max-width: 959px)")
    narrow = _block(CSS, "@media (max-width: 719px)")

    assert ".workbench-inspector" in inspector_overlay
    assert "grid-template-columns: minmax(10rem, 12rem) minmax(0, 1fr);" in inspector_overlay
    assert "position: fixed;" in inspector_overlay
    assert "inset: 0 0 0 auto;" in inspector_overlay
    assert "width: min(28rem, 90vw);" in inspector_overlay
    assert ".workbench-layout" in compact
    assert "grid-template-columns: minmax(0, 1fr);" in compact
    assert ".workspace-nav" in compact
    assert "flex-direction: row;" in compact
    assert ".workbench-inspector" in compact
    assert ".workbench-drawer > [role=\"dialog\"]" in compact
    assert "width: 100%;" in _block(compact, ".workbench-inspector,")
    assert ".product-grid" in narrow
    assert ".result-grid" in narrow
    assert '[aria-label="Enabled products"]' in narrow
    assert "grid-template-columns: minmax(0, 1fr);" in narrow


def test_only_the_graph_viewport_may_scroll_horizontally():
    graph = _block(CSS, ".graph-viewport")

    assert "overflow-x: auto;" in graph
    assert CSS.count("overflow-x: auto;") == 1
    assert "overflow-x: hidden;" in CSS


def test_forced_colours_keep_the_editor_focus_indicator_visible():
    forced = _block(CSS, "@media (forced-colors: active)")

    assert ":focus-visible" in forced
    assert "summary" in forced
    assert "outline-color: Highlight;" in forced
    assert "forced-color-adjust: auto;" in forced


def test_forced_colours_keep_status_chip_boundaries_visible():
    forced = _block(CSS, "@media (forced-colors: active)")

    assert ".rheplicant-editor .status-chip" in forced
    chip = _block(forced, ".rheplicant-editor .status-chip")
    assert "border-color: CanvasText;" in chip
    assert "forced-color-adjust: auto;" in chip


def test_inactive_graph_labels_keep_aa_contrast_without_opacity():
    assert ".dim{opacity:1}" in _SVG_STYLE
    assert ".dim rect,.dim line{stroke-dasharray:3 3}" in _SVG_STYLE
    assert ".dim text{font-style:italic}" in _SVG_STYLE
    assert "opacity:.22" not in _SVG_STYLE

    for theme, canvas in (("light", "#ffffff"), ("dark", "#111820")):
        for kind in ("source", "transform", "processing"):
            text_colour = _THEMES[theme][kind][2]
            assert _contrast(text_colour, canvas) >= 4.5
