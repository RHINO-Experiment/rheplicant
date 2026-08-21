from __future__ import annotations

import pytest
import yaml

from .test_document import BASE


@pytest.fixture
def candidate():
    panel = pytest.importorskip("panel")
    from tools.config_gui_spike import panel_app

    return panel, panel_app


def test_panel_adapter_uses_the_shared_yaml_transform(candidate):
    _, panel_app = candidate
    found = panel_app.apply_node_edit(
        BASE,
        "gain",
        enabled=True,
        settings={"type": "GainOperator", "gain": 2.0},
    )
    assert yaml.safe_load(found.yaml_text)["model"]["gain"]["gain"] == 2.0


def test_panel_adapter_delegates_to_the_shared_document_engine(candidate, monkeypatch):
    _, panel_app = candidate
    marker = object()
    received = {}

    def fake_set_node(yaml_text, node_id, *, enabled, settings):
        received.update(
            yaml_text=yaml_text,
            node_id=node_id,
            enabled=enabled,
            settings=settings,
        )
        return marker

    monkeypatch.setattr(panel_app, "set_node", fake_set_node)
    found = panel_app.apply_node_edit(
        BASE,
        "gain",
        enabled=True,
        settings={"type": "GainOperator", "gain": 2.0},
    )
    assert found is marker
    assert received == {
        "yaml_text": BASE,
        "node_id": "gain",
        "enabled": True,
        "settings": {"type": "GainOperator", "gain": 2.0},
    }


def test_panel_canvas_delegates_click_and_hover_by_stable_node_id(candidate):
    _, panel_app = candidate
    scripts = "\n".join(panel_app.SignalCanvas._scripts.values())
    assert "closest('[data-node-id]')" in scripts
    assert "data-node-id" in scripts
    assert "data-node-kind" in scripts
    assert "mouseover" in panel_app.SignalCanvas._template
    assert "onclick" in panel_app.SignalCanvas._template


def test_panel_candidate_builds_without_starting_a_server(candidate):
    panel, panel_app = candidate
    app = panel_app.build_app(BASE)
    assert isinstance(app, panel.viewable.Viewable)
    assert app.name == "Rheplicant config editor — Panel spike"
    labels = {button.label for button in app.select(panel.widgets.Button)}
    assert {"Previous node", "Next node", "Apply node edit"} <= labels


def test_panel_disables_the_gain_only_form_for_other_canvas_nodes(candidate):
    panel, panel_app = candidate
    app = panel_app.build_app(BASE)
    canvas = app.select(panel_app.SignalCanvas)[0]
    apply_button = next(
        button
        for button in app.select(panel.widgets.Button)
        if button.label == "Apply node edit"
    )
    canvas.click = "bandpass"
    assert apply_button.disabled
    canvas.click = "gain"
    assert not apply_button.disabled


def test_panel_yaml_load_refreshes_the_selected_node_projection(candidate):
    panel, panel_app = candidate
    app = panel_app.build_app(BASE)
    mirror = app.select(panel.widgets.TextAreaInput)[0]
    enabled = next(
        checkbox
        for checkbox in app.select(panel.widgets.Checkbox)
        if checkbox.name == "Node enabled"
    )
    load_button = next(
        button
        for button in app.select(panel.widgets.Button)
        if button.label == "Load YAML mirror"
    )
    mirror.value = "model: {}\n"
    load_button.clicks += 1
    assert not enabled.value
