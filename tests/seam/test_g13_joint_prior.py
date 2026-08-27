"""G13 through the seam: a declared joint prior, and what it does to the potential.

``tests/inference/test_graph_bridge.py`` pins the STRUCTURE of the crossing --
the graph carries the declaration, the covered latents are flat, an explicit
``rank_rtol`` survives. It cannot pin the arithmetic, because
``JeffreysPrior.information`` refuses ambient float32 by name and that file is
float32 on purpose. So the number lives here.

**The number is worth stating before the code.** For a bare power law
``mu = A (nu/nu0)^-beta`` over the block ``(log A, beta)`` under a radiometer
declaration ``sigma = f |mu|``, the Jeffreys prior is EXACTLY FLAT, and its
half-log-determinant on the 8x8 grid below is ``+15.80169853`` at every point
of a grid spanning two decades of amplitude and a unit of spectral index. Both
packages quote that constant in their own module docstrings and both derive it
in closed form: ``N^-1 = 1/(mu^2 f^2)`` while ``J_{k,i} = mu_k g_i(nu_k)``, so
every ``mu`` cancels and ``I_ij = (1 + 2 f^2) / f^2 * sum_k g_i g_j`` -- a
constant matrix in ``(log A, beta)``.

That makes it a genuine independent oracle rather than a pin: the test computes
it here in NumPy from the algebra, and separately asserts the potential this
seam produces moved by that much. A declaration that never reached the graph
moves it by zero; a declaration that reached it wearing the wrong block moves
it by something else.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from bayesmith.graph.evaluate import log_joint

from rheplicant import Coordinates, State
from rheplicant.inference import Bind, JeffreysPrior, Latent, ParameterSpace
from rheplicant.inference.graph_bridge import to_graph
from rheplicant.inference.noise import RadiometerNoise
from rheplicant.inference.uncertainty import as_noise_model
from rheplicant.radio import ForegroundOperator, assemble

N_TIME, N_FREQ = 8, 8
NU0 = 70e6
#: ``1 / sqrt(channel_width * integration_time)`` for the declaration below.
F = 1e-3

#: The design phase's grid: two decades of amplitude, a unit of spectral index.
GRID = [(la, be) for la in (6.8, 7.8, 8.8) for be in (2.05, 2.55, 3.05)]

#: The closed form, quoted to the digit both packages print.
FLAT_CONSTANT = 15.80169853


def _closed_form() -> float:
    """``0.5 log det I`` in NumPy, from the algebra and nothing else.

    ``g_0 = dlog mu/dlog A = 1`` and ``g_1 = dlog mu/dbeta = -log(nu/nu0)``;
    the ``1/f^2`` is the likelihood's weighting and the ``+2`` is the
    information the variance carries when sigma tracks the prediction. Every
    ``mu`` has cancelled, which is why no amplitude and no index appear.
    """
    nu = np.linspace(60e6, 85e6, N_FREQ) / NU0
    g = np.stack([np.ones(N_FREQ), -np.log(nu)])          # (2, N_FREQ)
    gram = (g @ g.T) * N_TIME                             # summed over time too
    information = (1.0 + 2.0 * F**2) / F**2 * gram
    return float(0.5 * np.linalg.slogdet(information)[1])


@pytest.fixture
def power_law():
    """The 8x8 power law, its data, its radiometer noise -- built once."""
    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    state = State(
        coords=Coordinates(time=jnp.arange(float(N_TIME)), freq=freq),
        key=jax.random.key(0),
        meta={"telescope": "g13-seam"},
    )
    twin = assemble(
        ForegroundOperator(
            amplitude=jnp.exp(jnp.array(7.8)),
            spectral_index=jnp.array(2.55),
            ref_freq=NU0,
        )
    )
    observed = twin(state).data
    noise = as_noise_model(
        RadiometerNoise(channel_width=1e6, integration_time=1.0),
        None,
        prediction_shape=jnp.shape(observed),
        caller="tests/seam/test_g13_joint_prior.py",
    )
    return {"twin": twin, "state": state, "observed": observed, "noise": noise}


def _bindings():
    return [
        Bind("fg_log_amp", into=lambda p: p["foregrounds"].amplitude, fn=jnp.exp),
        Bind("fg_beta", into=lambda p: p["foregrounds"].spectral_index),
    ]


def _covered_space(**overrides):
    return ParameterSpace(
        latents=[
            Latent("fg_log_amp", init=jnp.array(7.8)),
            Latent("fg_beta", init=jnp.array(2.55)),
        ],
        bindings=_bindings(),
        joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta"), **overrides),
    )


def _flat_space():
    """The same model with the block declared flat BY HAND and no joint prior.

    The control for the whole file: identical nodes, identical shapes,
    identical dtypes, and a posterior that is only the likelihood. This is what
    a forgotten declaration produces, and nothing about it looks wrong.
    """
    improper = dist.ImproperUniform(dist.constraints.real, (), event_shape=())
    return ParameterSpace(
        latents=[
            Latent("fg_log_amp", init=jnp.array(7.8), prior=improper),
            Latent("fg_beta", init=jnp.array(2.55), prior=improper),
        ],
        bindings=_bindings(),
    )


class TestTheDeclaredFactorIsTheJeffreysPrior:
    def test_the_closed_form_is_the_constant_both_packages_quote(self):
        """The oracle first, before it is used to judge anything.

        If this drifts, every assertion below is measuring the drift rather
        than the seam -- and the constant is quoted in two module docstrings
        and one migration page, so its going stale would be silent in three
        places at once.
        """
        assert _closed_form() == pytest.approx(FLAT_CONSTANT, abs=5e-9)

    @pytest.mark.parametrize("log_amp,beta", GRID)
    def test_the_potential_moves_by_the_closed_form_at_every_grid_point(
        self, power_law, log_amp, beta
    ):
        """Nine points, one number: the prior is flat, and it is THERE.

        Two graphs identical but for the declaration. Their ``log_joint``
        difference is the factor and nothing else -- the likelihood term, the
        flat sites and the data are shared by construction, so nothing else
        can contribute to it.
        """
        values = {"fg_log_amp": jnp.array(log_amp), "fg_beta": jnp.array(beta)}
        with_prior = to_graph(
            _covered_space(), power_law["twin"], power_law["state"],
            power_law["observed"], power_law["noise"],
        )
        without = to_graph(
            _flat_space(), power_law["twin"], power_law["state"],
            power_law["observed"], power_law["noise"],
        )
        assert without.joint_prior is None, "the control declared a joint prior"
        moved = float(log_joint(with_prior, values)) - float(log_joint(without, values))
        assert moved == pytest.approx(_closed_form(), rel=1e-9), (
            f"at (log A, beta) = ({log_amp}, {beta}) the declared factor moved "
            f"the potential by {moved:+.9f}; the closed form says "
            f"{_closed_form():+.9f}. A declaration that never reached the graph "
            "moves it by 0.0."
        )

    def test_the_prior_really_is_flat_across_the_grid(self, power_law):
        """The sibling assertion, and it is not decoration.

        Every parametrized case above compares against the SAME number, so a
        seam that ignored ``values`` entirely -- returning one cached factor --
        would satisfy all nine. This says the flatness is the model's, by
        checking that the likelihood term the same nine points produce is NOT
        constant: the difference is flat because the prior is, not because
        nothing varies.
        """
        without = to_graph(
            _flat_space(), power_law["twin"], power_law["state"],
            power_law["observed"], power_law["noise"],
        )
        likelihoods = [
            float(log_joint(without, {"fg_log_amp": jnp.array(la),
                                      "fg_beta": jnp.array(be)}))
            for la, be in GRID
        ]
        assert max(likelihoods) - min(likelihoods) > 1.0, (
            "the likelihood is constant across the grid too, so the flat "
            f"difference says nothing: spread {max(likelihoods) - min(likelihoods):.3e}"
        )

    def test_a_block_named_in_the_other_order_is_the_same_prior(self, power_law):
        """``over=`` reversed: a symmetric permutation, so the determinant is
        untouched and the potential must not move.

        Worth a test because the row ORDER of the information matrix is not
        untouched -- the two packages disagree about it, and that disagreement
        is registered. What this pins is that the disagreement stops at the
        rows and never reaches the density.
        """
        values = {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)}
        forward = to_graph(
            _covered_space(), power_law["twin"], power_law["state"],
            power_law["observed"], power_law["noise"],
        )
        reversed_space = ParameterSpace(
            latents=list(_covered_space().latents),
            bindings=_bindings(),
            joint_prior=JeffreysPrior(over=("fg_beta", "fg_log_amp")),
        )
        backward = to_graph(
            reversed_space, power_law["twin"], power_law["state"],
            power_law["observed"], power_law["noise"],
        )
        assert float(log_joint(forward, values)) == pytest.approx(
            float(log_joint(backward, values)), rel=1e-12
        )
