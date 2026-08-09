"""The `snapshot` node: a first-class place to preserve raw data.

SnapshotOperator deliberately declares no graph_node -- it lives in
rheplicant.core, which may not name a radio node. The node exists so that
At("snapshot", ...) addresses the snapshot by its own name, and so that a
taken snapshot is visible in the rendered signal path.
"""

import jax.numpy as jnp
import pytest

from rheplicant import At, Coordinates, State
from rheplicant.core.operator import SnapshotOperator
from rheplicant.radio import ADCOperator, FlaggingOperator, assemble
from rheplicant.radio.graph import RADIO_GRAPH

N_TIME, N_FREQ = 4, 3


@pytest.fixture
def state():
    return State(
        coords=Coordinates(
            time=jnp.arange(float(N_TIME)),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        data=jnp.ones((N_TIME, N_FREQ)),
    )


def test_snapshot_is_a_node_of_the_canonical_graph():
    assert "snapshot" in RADIO_GRAPH.nodes


def test_snapshot_sits_between_adc_and_flagging():
    assert ("adc", "snapshot") in RADIO_GRAPH.edges
    assert ("snapshot", "flagging") in RADIO_GRAPH.edges
    assert ("adc", "flagging") not in RADIO_GRAPH.edges


def test_snapshot_is_in_the_processing_segment():
    assert RADIO_GRAPH.nodes["snapshot"].segment == "processing"


def test_snapshot_is_a_transform():
    assert RADIO_GRAPH.nodes["snapshot"].kind == "transform"


def test_an_assembly_without_a_snapshot_skips_it(state):
    twin = assemble(ADCOperator(scale=jnp.array(1.0), n_bits=14),
                    FlaggingOperator(threshold=1e9))
    assert "snapshot" not in twin.lit
    assert "snapshot" in twin.skipped
    assert twin(state).data.shape == (N_TIME, N_FREQ)


def test_a_snapshot_placed_by_name_lights_that_node(state):
    twin = assemble(ADCOperator(scale=jnp.array(1.0), n_bits=14),
                    At("snapshot", SnapshotOperator(name="raw")),
                    FlaggingOperator(threshold=1e9))
    assert "snapshot" in twin.lit
    assert "snapshot/raw" in twin(state).aux


def test_the_snapshot_holds_the_data_as_it_was_before_flagging(state):
    twin = assemble(At("snapshot", SnapshotOperator(name="raw")),
                    FlaggingOperator(threshold=1e9))
    out = twin(state)
    assert jnp.array_equal(out.aux["snapshot/raw"], state.data)


def test_snapshot_operator_still_declares_no_graph_node():
    """The core class must not name a radio node -- see tests/core/test_layering.py."""
    assert getattr(SnapshotOperator, "graph_node", None) is None
