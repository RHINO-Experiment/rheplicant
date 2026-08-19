"""Panel half of the Config Plan 5 stack spike.

Run with::

    panel serve tools/config_gui_spike/panel_app.py --show

The app deliberately owns no document mutation.  Its callbacks translate
widgets to :mod:`rheplicant.gui` calls and render the returned snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping

import panel as pn
import param

from rheplicant.config import ConfigError
from rheplicant.gui import EditorSnapshot, replace_yaml, set_node, snapshot

pn.extension()

DEFAULT_YAML = """\
runtime:
  jax_enable_x64: true
model:
  gain:
    type: GainOperator
    gain: 1.0
runs:
  - name: forward
    kind: forward
"""


class SignalCanvas(pn.reactive.ReactiveHTML):
    """Clickable/hoverable projection of the shared SVG renderer."""

    value = param.String(default="")
    click = param.String(default="")
    hover = param.String(default="")

    _child_config = {"value": "literal"}
    _template = """
    <div id="container" class="signal-canvas"
      onclick="${script('click_handler')}"
      onmouseover="${script('hover_handler')}"
      onkeydown="${script('key_handler')}">
      {{ value }}
    </div>
    """
    _stylesheets = [
        """
        .signal-canvas {height: 100%; width: 100%; overflow: auto;}
        .signal-canvas svg {height: auto; max-height: 72vh; width: 100%;}
        .signal-canvas [role='button'] {cursor: pointer;}
        .signal-canvas [role='button']:hover,
        .signal-canvas [role='button']:focus {filter: brightness(1.12); outline: none;}
        """
    ]
    _scripts = {
        "click_handler": """
          const node = state.event.target.closest('[data-node-id]')
          if (node && node.getAttribute('role') === 'button') {
            data.click = node.getAttribute('data-node-id')
          }
        """,
        "hover_handler": """
          const node = state.event.target.closest('[data-node-id]')
          data.hover = node ? node.getAttribute('data-node-id') : ''
          if (node && node.getAttribute('data-node-kind')) {
            container.dataset.hoverKind = node.getAttribute('data-node-kind')
          }
        """,
        "key_handler": """
          if (state.event.key === 'Enter' || state.event.key === ' ') {
            const node = state.event.target.closest('[data-node-id]')
            if (node && node.getAttribute('role') === 'button') {
              state.event.preventDefault()
              data.click = node.getAttribute('data-node-id')
            }
          }
        """,
    }


def apply_node_edit(
    yaml_text: str,
    node_id: str,
    *,
    enabled: bool,
    settings: Mapping[str, object] | None = None,
) -> EditorSnapshot:
    """The adapter seam used by the Panel callback and parity test."""
    return set_node(yaml_text, node_id, enabled=enabled, settings=settings)


def build_app(initial_yaml: str = DEFAULT_YAML) -> pn.Column:
    """Construct the candidate without starting a server or browser."""
    current = snapshot(initial_yaml)
    canvas = SignalCanvas(value=current.svg, min_width=520, sizing_mode="stretch_both")
    yaml_mirror = pn.widgets.TextAreaInput(
        label="YAML (source of truth)",
        value=current.yaml_text,
        height=260,
        sizing_mode="stretch_width",
    )
    selected = pn.widgets.StaticText(label="Selected node", value="gain")
    hovered = pn.widgets.StaticText(label="Hovered node", value="")
    enabled = pn.widgets.Checkbox(label="Node enabled", value=True)
    operator_type = pn.widgets.TextInput(label="type", value="GainOperator")
    gain = pn.widgets.FloatInput(label="gain", value=1.0, step=0.05)
    status = pn.pane.Alert("Ready", alert_type="light")
    thin_slice_note = pn.widgets.StaticText(
        value="Task 1 edits gain only; other nodes prove canvas navigation."
    )
    apply_button = pn.widgets.Button(label="Apply node edit", color="primary")
    load_button = pn.widgets.Button(label="Load YAML mirror")
    previous_button = pn.widgets.Button(label="Previous node")
    next_button = pn.widgets.Button(label="Next node")

    def show(found: EditorSnapshot) -> None:
        canvas.value = found.svg
        yaml_mirror.value = found.yaml_text
        select(selected.value)
        status.object = "YAML transformed by rheplicant.gui"
        status.alert_type = "success"

    def select(node_id: str) -> None:
        selected.value = node_id
        card = next((node for node in current.nodes if node.node_id == node_id), None)
        if card is not None:
            enabled.value = card.lit
        thin_slice_editable = node_id == "gain"
        enabled.disabled = not thin_slice_editable
        operator_type.disabled = not thin_slice_editable
        gain.disabled = not thin_slice_editable
        apply_button.disabled = not thin_slice_editable

    def choose(event: param.parameterized.Event) -> None:
        select(str(event.new))

    def walk(step: int) -> None:
        editable = [
            node_id
            for node_id in current.walk_order
            if next(node for node in current.nodes if node.node_id == node_id).editable
        ]
        index = editable.index(selected.value) if selected.value in editable else 0
        select(editable[(index + step) % len(editable)])

    def hover(event: param.parameterized.Event) -> None:
        hovered.value = str(event.new)

    def apply(_event: object) -> None:
        nonlocal current
        settings = (
            {"type": operator_type.value, "gain": gain.value}
            if enabled.value
            else None
        )
        try:
            current = apply_node_edit(
                yaml_mirror.value,
                selected.value,
                enabled=enabled.value,
                settings=settings,
            )
            show(current)
        except ConfigError as error:  # Panel displays the layer's exact refusal.
            status.object = str(error)
            status.alert_type = "danger"

    def load(_event: object) -> None:
        nonlocal current
        try:
            current = replace_yaml(yaml_mirror.value)
            show(current)
        except ConfigError as error:
            status.object = str(error)
            status.alert_type = "danger"

    canvas.param.watch(choose, "click")
    canvas.param.watch(hover, "hover")
    apply_button.on_click(apply)
    load_button.on_click(load)
    previous_button.on_click(lambda _event: walk(-1))
    next_button.on_click(lambda _event: walk(1))

    settings = pn.Card(
        selected,
        hovered,
        enabled,
        operator_type,
        gain,
        thin_slice_note,
        pn.Row(previous_button, next_button),
        apply_button,
        title="Node settings",
        collapsed=False,
    )
    app = pn.Column(
        "# Rheplicant config editor — Panel spike",
        pn.Row(canvas, pn.Column(settings, yaml_mirror, load_button, status)),
        name="Rheplicant config editor — Panel spike",
        sizing_mode="stretch_both",
    )
    return app


if pn.state.served:  # pragma: no cover - exercised by ``panel serve``.
    build_app().servable()


__all__ = ["SignalCanvas", "apply_node_edit", "build_app"]
