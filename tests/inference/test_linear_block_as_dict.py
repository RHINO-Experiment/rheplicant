"""``LinearBlock.as_dict``: the one call that makes a solve's answer consumable.

A solve returns the block's OWN domain — a bare array for a ``name=`` block, a
``{name: array}`` dict for a ``names=`` group. Only the second is what anything
downstream reads. These tests pin both halves of that: that the wrap is
correct and idempotent across the two spellings, and that the consumers really
do reject the unwrapped form, so the helper cannot quietly stop earning its
place without a test going red.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    ParameterSpace,
    identifiability,
    linear_operator,
    wiener_solve,
)
from rheplicant.inference.engines import conditional_potential
from rheplicant.radio import GainOperator, SkyOperator

SKY, GAIN, TRUE_GAIN, NOISE = 100.0, 1.0, 1.1, 0.5

#: Loose enough that the data drives the solve. A grouped block takes one per
#: member, which is the asymmetry the two spellings carry all the way down.
PRIOR = {"gain": 10.0}


@pytest.fixture
def twin():
    return Pipeline(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.array(GAIN)),
        names=("sky", "gain"),
    )


@pytest.fixture
def space():
    return ParameterSpace.direct(
        "gain", init=GAIN, into=lambda p: p["gain"].gain, linear=True
    )


@pytest.fixture
def observed(space, twin, template_state):
    forward, _ = space.forward_fn(twin, template_state)
    return forward({"gain": jnp.array(TRUE_GAIN)})


class TestAsDict:
    def test_wraps_the_bare_answer_of_a_singular_block(
        self, space, twin, template_state, observed
    ):
        block = linear_operator(space, twin, template_state, "gain")
        solved, _ = wiener_solve(block, observed, noise_std=NOISE, prior_std=10.0)
        assert not isinstance(solved, dict)
        assert block.as_dict(solved) == {"gain": solved}

    def test_is_a_no_op_on_a_group(self, space, twin, template_state, observed):
        block = linear_operator(space, twin, template_state, names=("gain",))
        solved, _ = wiener_solve(block, observed, noise_std=NOISE, prior_std=PRIOR)
        assert block.as_dict(solved) == solved

    def test_the_two_spellings_agree_once_wrapped(
        self, space, twin, template_state, observed
    ):
        """The whole point: one call, correct whichever way the block was built."""
        singular = linear_operator(space, twin, template_state, "gain")
        grouped = linear_operator(space, twin, template_state, names=("gain",))
        one, _ = wiener_solve(singular, observed, noise_std=NOISE, prior_std=10.0)
        many, _ = wiener_solve(grouped, observed, noise_std=NOISE, prior_std=PRIOR)
        assert singular.as_dict(one).keys() == grouped.as_dict(many).keys()
        assert jnp.allclose(singular.as_dict(one)["gain"], grouped.as_dict(many)["gain"])

    def test_does_not_mutate_or_alias_the_group_it_was_given(
        self, space, twin, template_state, observed
    ):
        block = linear_operator(space, twin, template_state, names=("gain",))
        solved, _ = wiener_solve(block, observed, noise_std=NOISE, prior_std=PRIOR)
        wrapped = block.as_dict(solved)
        wrapped["intruder"] = 1
        assert "intruder" not in solved

    def test_a_group_refuses_a_bare_array_by_name(self, space, twin, template_state):
        """The wrong block's answer is a caller error, not something to wrap."""
        block = linear_operator(space, twin, template_state, names=("gain",))
        with pytest.raises(ParameterSpaceError, match="groups \\['gain'\\]"):
            block.as_dict(jnp.array(1.0))

    def test_a_group_refuses_a_dict_keyed_wrong(self, space, twin, template_state):
        block = linear_operator(space, twin, template_state, names=("gain",))
        with pytest.raises(ParameterSpaceError, match="keyed by \\['sky'\\]"):
            block.as_dict({"sky": jnp.array(1.0)})


class TestWhyTheWrapIsNeeded:
    """Pin that the bare form really is refused downstream.

    Six consumers, six different exceptions, and not one of them names the
    actual mistake. If a later change makes them accept the bare array, this
    class goes red and ``as_dict`` should be reconsidered rather than kept out
    of habit.
    """

    @pytest.fixture
    def bare(self, space, twin, template_state, observed):
        block = linear_operator(space, twin, template_state, "gain")
        solved, _ = wiener_solve(block, observed, noise_std=NOISE, prior_std=10.0)
        return block, solved

    def test_forward_fn_rejects_it(self, space, twin, template_state, bare):
        block, solved = bare
        forward, _ = space.forward_fn(twin, template_state)
        with pytest.raises(TypeError):
            forward(solved)
        assert jnp.shape(forward(block.as_dict(solved))) == jnp.shape(block.offset)

    def test_bind_rejects_it(self, space, twin, bare):
        block, solved = bare
        with pytest.raises(TypeError):
            space.bind(twin, solved)
        assert space.bind(twin, block.as_dict(solved)) is not twin

    def test_identifiability_at_rejects_it(self, space, twin, template_state, bare):
        block, solved = bare
        with pytest.raises(TypeError):
            identifiability(space, twin, template_state, at=solved)
        report = identifiability(space, twin, template_state, at=block.as_dict(solved))
        assert report.nullity == 0

    def test_linear_operator_at_rejects_it(self, space, twin, template_state, bare):
        block, solved = bare
        with pytest.raises(TypeError):
            linear_operator(space, twin, template_state, "gain", at=solved)
        rebuilt = linear_operator(
            space, twin, template_state, "gain", at=block.as_dict(solved)
        )
        assert rebuilt.name == "gain"

    def test_conditional_potential_rejects_it(
        self, space, twin, template_state, observed, bare
    ):
        from rheplicant.inference.engines import Conditioning
        from rheplicant.inference.noise import HomoscedasticNoise

        block, solved = bare
        forward, values0 = space.forward_fn(twin, template_state)
        cond = Conditioning(
            space, twin, template_state, observed,
            HomoscedasticNoise(jnp.array(NOISE)), forward,
        )
        potential = conditional_potential(cond, ("gain",), values0)
        with pytest.raises(TypeError):
            potential(solved)
        assert jnp.isfinite(potential(block.as_dict(solved)))

    def test_fisher_information_rejects_it(self, space, twin, template_state, bare):
        from rheplicant.inference.uncertainty import fisher_information

        block, solved = bare
        forward, _ = space.forward_fn(twin, template_state)
        with pytest.raises(TypeError):
            fisher_information(forward, solved, noise_std=NOISE)
        fisher = fisher_information(forward, block.as_dict(solved), noise_std=NOISE)
        assert fisher.matrix.shape == (1, 1)


def test_a_grouped_solve_round_trips_through_forward(space, twin, template_state, observed):
    """The recommended spelling needs no wrap at all — which is the point of it."""
    block = linear_operator(space, twin, template_state, names=("gain",))
    solved, _ = wiener_solve(block, observed, noise_std=NOISE, prior_std=PRIOR)
    forward, _ = space.forward_fn(twin, template_state)
    assert jnp.allclose(forward(solved), observed, atol=1e-3)
    assert jax.tree.structure(solved) == jax.tree.structure(block.as_dict(solved))
