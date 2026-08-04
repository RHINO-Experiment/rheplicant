"""The one enforced member of the operator declaration contract: ``requires``.

Every guard here is pinned on BOTH sides — a stage that declares the path is
found, a stage that does not is not — and the walk is pinned at depth, because
the failure that matters is a stochastic stage buried in a summed branch rather
than sitting at the top of a Pipeline.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.contract import (
    RANDOMNESS,
    describe_stages,
    stages_requiring,
    walk_operators,
)
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import State


class Quiet(AbstractOperator):
    """Deterministic source: value * ones(4)."""

    requires: ClassVar[tuple[str, ...]] = ("coords.freq",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    value: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(self.value * jnp.ones(4))


class Loud(AbstractOperator):
    """A source that draws randomness, and says so."""

    requires: ClassVar[tuple[str, ...]] = ("coords.freq", "key")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    scale: jax.Array

    def __call__(self, state: State) -> State:
        subkey, state = state.next_key()
        return state.with_data(self.scale * jax.random.normal(subkey, (4,)))


class Scale(AbstractOperator):
    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    factor: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(state.data * self.factor)


@pytest.fixture
def nested():
    """Loud sits two composites deep, and every scalar is distinct.

    2 and 3 are the summed sources, 5 the trunk factor, 7 the noise scale: no
    two of them can be swapped without moving a number this file asserts.
    """
    return Pipeline(
        SumOperator(
            Quiet(value=jnp.array(2.0)),
            Pipeline(
                Quiet(value=jnp.array(3.0)),
                Loud(scale=jnp.array(7.0)),
                names=("clean", "noisy"),
            ),
            names=("plain", "branch"),
        ),
        Scale(factor=jnp.array(5.0)),
        names=("t_ant", "trunk"),
    )


class TestWalk:
    def test_root_is_yielded_with_an_empty_label(self, nested):
        label, op = next(iter(walk_operators(nested)))
        assert label == ""
        assert op is nested

    def test_labels_are_the_slash_joined_stage_names(self, nested):
        labels = [label for label, _ in walk_operators(nested)]
        assert labels == [
            "",
            "t_ant",
            "t_ant/plain",
            "t_ant/branch",
            "t_ant/branch/clean",
            "t_ant/branch/noisy",
            "trunk",
        ]

    def test_every_operator_instance_is_reached(self, nested):
        kinds = sorted(type(op).__name__ for _, op in walk_operators(nested))
        assert kinds == ["Loud", "Pipeline", "Pipeline", "Quiet", "Quiet", "Scale", "SumOperator"]

    def test_a_bare_operator_walks_to_itself_alone(self):
        lone = Scale(factor=jnp.array(5.0))
        assert list(walk_operators(lone)) == [("", lone)]

    def test_an_unnamed_composite_contributes_no_label_segment(self, nested):
        """A wrapper that is not Pipeline/Sum/Select is spine, not a stage.

        `Assembly` is the shipped one: its `.operator` field holds the fold, and
        a refusal must quote the graph node id, not `operator/gain`.
        """

        class Wrapper(AbstractOperator):
            inner: AbstractOperator

            def __call__(self, state):
                return self.inner(state)

        wrapped = Wrapper(inner=nested)
        labels = [label for label, _ in walk_operators(wrapped)]
        assert labels[:3] == ["", "", "t_ant"]

    def test_selector_branches_are_walked_by_name(self):
        select = SelectOperator(
            Quiet(value=jnp.array(2.0)),
            Loud(scale=jnp.array(7.0)),
            names=("cold", "hot"),
            switch_key="sw",
        )
        assert [label for label, _ in walk_operators(select)] == ["", "cold", "hot"]


class TestStagesRequiring:
    def test_finds_the_nested_declaring_stage_only(self, nested):
        found = stages_requiring(nested, RANDOMNESS)
        assert [(label, type(op).__name__) for label, op in found] == [
            ("t_ant/branch/noisy", "Loud")
        ]

    def test_a_deterministic_tree_declares_nothing(self):
        clean = Pipeline(
            Quiet(value=jnp.array(2.0)), Scale(factor=jnp.array(5.0)), names=("sky", "trunk")
        )
        assert stages_requiring(clean, RANDOMNESS) == ()

    def test_other_paths_are_answerable_too(self, nested):
        found = stages_requiring(nested, "data")
        assert [label for label, _ in found] == ["trunk"]

    def test_the_root_itself_can_be_the_declaring_stage(self):
        lone = Loud(scale=jnp.array(7.0))
        assert stages_requiring(lone, RANDOMNESS) == (("", lone),)


class TestDescribeStages:
    def test_names_the_class_and_the_label(self, nested):
        assert describe_stages(stages_requiring(nested, RANDOMNESS)) == (
            "Loud at 't_ant/branch/noisy'"
        )

    def test_an_unlabelled_root_is_named_by_class_alone(self):
        lone = Loud(scale=jnp.array(7.0))
        assert describe_stages(stages_requiring(lone, RANDOMNESS)) == "Loud"


def test_the_declaration_matches_what_the_operator_actually_does():
    """The contract is only worth enforcing if it is true of the shipped set.

    Every operator in `src/` that touches `jax.random` reaches it through
    `State.next_key`, so a stage that draws randomness cannot run without a
    key — which is exactly what `'key' in requires` claims.
    """
    coords = Coordinates(time=jnp.arange(4.0), freq=jnp.linspace(60e6, 85e6, 4))
    keyless = State(coords=coords, key=None)
    from rheplicant.core.errors import MissingKeyError

    with pytest.raises(MissingKeyError):
        Loud(scale=jnp.array(7.0))(keyless)
    # ...and one that does not declare it runs on the same keyless state.
    assert Quiet(value=jnp.array(2.0))(keyless).data.shape == (4,)
