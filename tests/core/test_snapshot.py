"""Tests for raw-data preservation: State.checkpoint + SnapshotOperator."""

import jax.numpy as jnp
import pytest

from replicant import Pipeline, SnapshotOperator, State
from replicant.core.errors import StateValidationError
from replicant.core.operator import LambdaOperator


class TestCheckpoint:
    def test_snapshot_survives_overwrite(self):
        s = State(data=jnp.ones(3)).checkpoint("raw")
        s2 = s.with_data(jnp.zeros(3))
        assert jnp.array_equal(s2.aux["snapshot/raw"], jnp.ones(3))
        assert jnp.array_equal(s2.data, jnp.zeros(3))

    def test_default_name(self):
        s = State(data=jnp.ones(2)).checkpoint()
        assert "snapshot/raw" in s.aux

    def test_original_untouched(self):
        s = State(data=jnp.ones(2))
        s.checkpoint("x")
        assert "snapshot/x" not in s.aux

    def test_no_data_raises(self):
        with pytest.raises(StateValidationError, match="nothing to snapshot"):
            State().checkpoint()


class TestSnapshotOperator:
    def test_in_pipeline(self):
        pipe = Pipeline(
            SnapshotOperator(name="raw"),
            LambdaOperator.on_data(lambda d: d * 0.0),
            names=("snap", "destroy"),
        )
        out = pipe(State(data=jnp.full(3, 7.0)))
        assert jnp.all(out.data == 0.0)
        assert jnp.array_equal(out.aux["snapshot/raw"], jnp.full(3, 7.0))
