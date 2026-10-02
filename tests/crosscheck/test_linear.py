"""``rheplicant.inference.linear`` against ``bayesmith.exact``.

The solve names in ``rheplicant.inference.linear`` delegate to bayesmith, so
comparing the two solves would compare bayesmith with itself. What this
package still owns is ``linear_operator`` and ``check_linearity``, the
affinity criterion, and the refusals its facade raises before the seam. Those
are the comparisons here:

* the linearity verdict: each conditional of a bilinear model is affine and
  the pair is not, on both sides;
* a rule for sigma is refused at the conjugate seam on both sides;
* a 1-D sigma whose axis the prediction cannot settle is refused here and
  cannot be refused at the seam, with the reason measured and the size of
  the error it would cause.

"near" is this package and "far" is bayesmith throughout.

Everything is built inside ``with jax.enable_x64(True):``, arrays included.
The context governs the operation and not the array, so a fixture built
outside it arrives as float32 in a float64 graph. Measured when this file was
written: that made ``check_linearity`` report unresolved departures, and the
two sides agreed bitwise once the arrays moved inside.

Moved from bayesmith's ``tests/crosscheck/`` at its ``d861220``, the last
revision that held it. Two tests that asserted bayesmith's own behaviour and
read nothing from this package stayed behind.
"""

from __future__ import annotations

# Module scope: ``from __future__ import annotations`` stringifies the operator
# classes' body annotations, and dataclasses resolves ``ClassVar`` against this
# module's globals.
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N_TIME, N_FREQ = 8, 8
TONE_CHANNEL = 3
TONE_KELVIN = 5000.0

#: Flat enough that the prior does not resolve the bilinear degeneracy: the
#: value ``tests/inference/test_degenerate_partition.py`` uses.
PRIOR_STD = 1e6
NOISE_STD = 1.0


def _arrays() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The shared ``gain x T_ant`` fixture. Call inside the x64 block."""

    def poly_basis(n: int, degree: int) -> jax.Array:
        x = jnp.linspace(-1.0, 1.0, n)
        return jnp.stack([x**k for k in range(degree)], axis=1)

    time_basis = poly_basis(N_TIME, 3)
    freq_basis = poly_basis(N_FREQ, 3)
    coeff0 = jnp.array([[3000.0, -180.0, 40.0], [120.0, 25.0, -8.0], [-45.0, 6.0, 2.0]])
    gain0 = 1.5 + 0.05 * jnp.arange(N_TIME, dtype=float)
    t_ant0 = time_basis @ coeff0 @ freq_basis.T
    tone = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(TONE_KELVIN)
    data = gain0[:, None] * (t_ant0 + tone[None, :])
    return gain0, t_ant0, tone, data


def _near_model(gain0, t_ant0, tone):
    """``(space, pipeline, template, at)`` in this package's vocabulary."""
    from rheplicant import Coordinates, State
    from rheplicant.core.operator import AbstractOperator
    from rheplicant.core.pipeline import Pipeline
    from rheplicant.inference import Bind, Latent, ParameterSpace
    from rheplicant.radio import GainOperator

    class AntennaTemperature(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
        provides: ClassVar[tuple[str, ...]] = ("data",)
        t_ant: jax.Array

        def __call__(self, state):
            return state.with_data(self.t_ant)

    class CalibrationTone(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("data",)
        provides: ClassVar[tuple[str, ...]] = ("data",)
        tone: jax.Array

        def __call__(self, state):
            return state.with_data(state.data + self.tone[None, :])

    pipeline = Pipeline(
        AntennaTemperature(t_ant=t_ant0),
        CalibrationTone(tone=tone),
        GainOperator(gain=gain0),
        names=("t_ant", "tone", "gain"),
    )
    # Both latents are declared linear: each conditional is affine and the
    # model is bilinear.
    space = ParameterSpace(
        latents=[
            Latent("gain", init=gain0, linear=True),
            Latent("t_ant", init=t_ant0, linear=True),
        ],
        bindings=[
            Bind("gain", into=lambda p: p["gain"].gain),
            Bind("t_ant", into=lambda p: p["t_ant"].t_ant),
        ],
    )
    template = State(
        data=jnp.zeros((N_TIME, N_FREQ)),
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 80e6, N_FREQ),
        ),
    )
    return space, pipeline, template, {"gain": gain0, "t_ant": t_ant0}


def _near_pieces(gain0, t_ant0, tone, block_name):
    """``(block, prior_mean, prior_std)``, through the checking entry point."""
    from rheplicant.inference.linear import linear_operator

    space, pipeline, template, at = _near_model(gain0, t_ant0, tone)
    block = linear_operator(space, pipeline, template, names=(block_name,), at=at)
    return (
        block,
        {block_name: jnp.zeros_like(at[block_name])},
        {block_name: jnp.full_like(at[block_name], PRIOR_STD)},
    )


def _far_graph(gain0, t_ant0, tone, data, sigma: Any = NOISE_STD):
    """The same model as a bayesmith graph.

    The linearity claim lives in a different place on each side: this package
    declares ``linear=True`` on the latent, bayesmith declares
    ``linear_in=(...)`` on the deterministic node.
    """
    import numpyro.distributions as dist
    from bayesmith import det, observe, sample, trace

    def model():
        gain = sample("gain", lambda: dist.Normal(jnp.zeros_like(gain0), PRIOR_STD).to_event(1))
        t_ant = sample("t_ant", lambda: dist.Normal(jnp.zeros_like(t_ant0), PRIOR_STD).to_event(2))
        pred = det(
            "pred",
            lambda g, t: g[:, None] * (t + tone[None, :]),
            gain,
            t_ant,
            linear_in=("gain", "t_ant"),
        )
        observe("d", lambda mu: dist.Normal(mu, sigma).to_event(2), pred, obs=data)

    return trace(model)


def _far_block(gain0, t_ant0, tone, data, block_name):
    """bayesmith's linear block for one latent, through its checking entry point."""
    from bayesmith.exact.linearity import linear_operator

    graph = _far_graph(gain0, t_ant0, tone, data)
    outside = {"gain": gain0, "t_ant": t_ant0}
    outside.pop(block_name)
    return linear_operator(graph, [block_name], outside)


def test_check_linearity_accepts_the_same_blocks_on_both_sides():
    """Each conditional is affine and the pair is not: the same verdict twice.

    A group holding both latents is what an alternating solve treats as one
    linear block, and is why a bilinear model needs more than one.
    """
    from bayesmith.errors import StructureError
    from bayesmith.exact.linearity import check_linearity as far

    from rheplicant.core.errors import ParameterSpaceError
    from rheplicant.inference.linear import check_linearity as near

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        graph = _far_graph(gain0, t_ant0, tone, data)
        space, pipeline, template, at = _near_model(gain0, t_ant0, tone)
        for name in ("gain", "t_ant"):
            outside = {"gain": gain0, "t_ant": t_ant0}
            outside.pop(name)
            far(graph, [name], outside)
            near(space, pipeline, template, names=(name,), at=at)
        with pytest.raises(StructureError):
            far(graph, ["gain", "t_ant"], {})
        with pytest.raises(ParameterSpaceError):
            near(space, pipeline, template, names=("gain", "t_ant"), at=at)


def _refusal(fn) -> str:
    """Run ``fn`` and return the refusal text, or ``''`` if it returned."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- the class is what is under test
        return f"{type(exc).__name__}: {exc}"
    return ""


@pytest.mark.parametrize("depends_on_prediction", [False, True])
def test_a_rule_for_sigma_is_refused_at_the_conjugate_seam_on_both_sides(
    depends_on_prediction,
):
    """A conjugate solve needs a decided covariance, not a rule for one.

    This package refuses a ``NoiseModel`` by name, with a longer sentence when
    the model depends on the prediction: the solve has no prediction to
    evaluate the rule at, the prediction being what it solves for.

    bayesmith's ``precision=`` takes a ``Precision``, an operator, so a rule is
    refused by the protocol. Both refuse; what is compared is that neither
    freezes sigma somewhere without saying so.
    """
    from bayesmith.exact.solve import wiener_solve as far

    from rheplicant.inference.linear import wiener_solve as near
    from rheplicant.inference.noise import HomoscedasticNoise, RadiometerNoise

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        block, prior_mean, prior_std = _near_pieces(gain0, t_ant0, tone, "gain")
        model = (
            RadiometerNoise(channel_width=1e6, integration_time=1.0, floor=1.0)
            if depends_on_prediction
            else HomoscedasticNoise(sigma=NOISE_STD)
        )
        near_text = _refusal(
            lambda: near(
                block,
                data,
                noise_std=model,
                prior_mean=prior_mean,
                prior_std=prior_std,
                require_convergence=None,
            )
        )
        far_block = _far_block(gain0, t_ant0, tone, data, "gain")
        far_text = _refusal(
            lambda: far(
                far_block,
                precision={"d": (lambda mu: 0.05 * mu)},
                require_convergence=None,
            )
        )
    assert "ParameterSpaceError" in near_text, near_text[:300]
    if depends_on_prediction:
        assert "has no prediction to evaluate it at" in near_text
    else:
        assert "takes a plain sigma array" in near_text
    assert far_text, "bayesmith accepted a callable where a Precision is required"


def test_the_ambiguous_1d_sigma_is_resolved_before_bayesmith_can_see_it():
    """Why the 1-D sigma refusal lives in this package's facade.

    This package refuses a 1-D ``noise_std`` whose axis the prediction cannot
    settle. Against an ``(8, 8)`` grid a length-8 vector reads as one sigma
    per time sample or as one per frequency channel, NumPy picks the trailing
    axis, and every downstream number is finite and correctly shaped.

    bayesmith cannot state that refusal at the seam, because of numpyro:
    ``dist.Normal(loc, scale)`` runs ``promote_shapes`` in its constructor,
    inside the user's ``dist_fn``, so by the time bayesmith reads
    ``distribution.scale`` a bare ``(8,)`` is already ``(1, 8)`` and cannot be
    told from an explicit ``(1, 8)``. A guard there would miss the ambiguous
    case or refuse the unambiguous one.

    If numpyro stops promoting, the first assertion fails, and the guard
    becomes writable on the far side.
    """
    import numpyro.distributions as dist
    from bayesmith.exact.gaussian import noise_std_at

    from rheplicant.inference.linear import wiener_solve as near

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        ambiguous = jnp.linspace(0.01, 1.0, N_TIME)
        # 1. The declarations are already identical at the distribution.
        assert jnp.shape(dist.Normal(data, ambiguous).scale) == (1, N_FREQ)
        assert jnp.shape(dist.Normal(data, ambiguous[None, :]).scale) == (1, N_FREQ)
        assert jnp.shape(dist.Normal(data, ambiguous[:, None]).scale) == (N_TIME, 1)

        # 2. So the graph reads the bare vector as per-frequency.
        at = {"gain": gain0, "t_ant": t_ant0}
        bare = noise_std_at(_far_graph(gain0, t_ant0, tone, data, ambiguous), at)
        per_freq = noise_std_at(_far_graph(gain0, t_ant0, tone, data, ambiguous[None, :]), at)
        per_time = noise_std_at(_far_graph(gain0, t_ant0, tone, data, ambiguous[:, None]), at)
        assert np.array_equal(np.asarray(bare["d"]), np.asarray(per_freq["d"]))
        assert not np.array_equal(np.asarray(bare["d"]), np.asarray(per_time["d"]))

        # 3. And this package refuses the same declaration.
        block, prior_mean, prior_std = _near_pieces(gain0, t_ant0, tone, "gain")
        text = _refusal(
            lambda: near(
                block,
                data,
                noise_std=ambiguous,
                prior_mean=prior_mean,
                prior_std=prior_std,
                require_convergence=None,
            )
        )
    assert "StateValidationError" in text, text[:300]
    assert "more than one legitimate reading" in text


def test_the_silent_axis_choice_is_over_confident_and_by_how_much():
    """The size and sign of the error the refusal above prevents.

    Posterior standard deviations of the eight ``gain`` samples, from a dense
    solve, under the two readings of the same length-8 vector
    ``linspace(0.01, 1.0, 8)``:

    ======================================  ==========  ==========  =======
    reading                                 min         max         spread
    ======================================  ==========  ==========  =======
    per-time ``(8, 1)``, what was meant     9.165e-07   8.690e-05   94.8x
    per-freq ``(1, 8)``, what is taken      3.055e-06   3.228e-06    1.1x
    ======================================  ==========  ==========  =======

    The sample the data constrains worst comes back 26.9x narrower than it
    is, and the 95x structure the sigma vector describes is averaged away.

    The assertions are bounds on the two spreads and on their ratio, so they
    survive a change in the fixture's numbers and fail if the effect reverses
    or disappears.
    """
    with jax.enable_x64(True):
        _, t_ant0, tone, _ = _arrays()
        sigma = np.linspace(0.01, 1.0, N_TIME)

        def mu(x):
            return np.ravel(np.asarray(x[:, None] * (t_ant0 + tone[None, :])))

        offset = mu(jnp.zeros(N_TIME))
        design = np.stack(
            [mu(jnp.zeros(N_TIME).at[i].set(1.0)) - offset for i in range(N_TIME)],
            axis=1,
        )
        spreads = {}
        for label, grid in (
            ("per_time", np.broadcast_to(sigma[:, None], (N_TIME, N_FREQ))),
            ("per_freq", np.broadcast_to(sigma[None, :], (N_TIME, N_FREQ))),
        ):
            inverse_noise = np.diag(1.0 / np.ravel(grid) ** 2)
            normal = design.T @ inverse_noise @ design + np.eye(N_TIME) / PRIOR_STD**2
            spreads[label] = np.sqrt(np.diag(np.linalg.inv(normal)))

    assert spreads["per_time"].max() / spreads["per_time"].min() > 50.0
    assert spreads["per_freq"].max() / spreads["per_freq"].min() < 2.0
    # What is taken is narrower than what was meant, on the worst-constrained
    # sample.
    assert spreads["per_time"].max() / spreads["per_freq"].max() > 20.0
