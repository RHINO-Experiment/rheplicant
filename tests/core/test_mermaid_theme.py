"""to_mermaid takes a theme, like to_svg and to_html already do."""

import jax.numpy as jnp
import pytest

from rheplicant.radio import GainOperator, SkyOperator, assemble
from rheplicant.radio.graph import RADIO_GRAPH


@pytest.fixture
def twin():
    return assemble(SkyOperator(amplitude=jnp.array(1.0)),
                    GainOperator(gain=jnp.array(1.0)))


def test_signal_graph_default_theme_is_light():
    assert RADIO_GRAPH.to_mermaid() == RADIO_GRAPH.to_mermaid(theme="light")


def test_signal_graph_dark_differs_from_light():
    assert RADIO_GRAPH.to_mermaid(theme="dark") != RADIO_GRAPH.to_mermaid(theme="light")


def test_signal_graph_refuses_an_unknown_theme():
    with pytest.raises(KeyError):
        RADIO_GRAPH.to_mermaid(theme="solarized")


def test_assembly_default_theme_is_light(twin):
    assert twin.to_mermaid() == twin.to_mermaid(theme="light")


def test_assembly_dark_differs_from_light(twin):
    assert twin.to_mermaid(theme="dark") != twin.to_mermaid(theme="light")


def test_assembly_dark_lights_the_same_nodes(twin):
    """Theming changes colour, never which nodes are lit."""
    light, dark = twin.to_mermaid(theme="light"), twin.to_mermaid(theme="dark")
    for node in twin.lit:
        assert node in light and node in dark


def test_light_output_is_unchanged_by_the_theme_parameter():
    """The whole point: adding the knob must not move the default rendering.

    These are the three classDef lines the renderer emitted before the theme
    parameter existed, in the order it emitted them.
    """
    out = RADIO_GRAPH.to_mermaid()
    assert "classDef lit fill:#FAC775,stroke:#854F0B,color:#412402;" in out
    assert "classDef wire fill:#F1EFE8,stroke:#854F0B,color:#444441;" in out
    assert "classDef dim fill:#F1EFE8,stroke:#B4B2A9,color:#B4B2A9;" in out
