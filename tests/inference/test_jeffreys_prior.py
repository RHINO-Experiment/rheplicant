"""``JeffreysPrior`` — available, off by default, and honest about what it is.

The headline is a measurement rather than a warning: under
:class:`~rheplicant.inference.noise.RadiometerNoise`, which is this package's
DEFAULT noise model, the Jeffreys prior of a bare power law over
``(log A, beta)`` **is the flat prior**. Nine grid points spanning two decades
in amplitude and a full unit in spectral index return the same
half-log-determinant to the last printed digit, and its gradient there is
``(0.0, 5.55e-17)``. Turning it on there costs a Jacobian per leapfrog step and
buys a constant.

Under :class:`~rheplicant.inference.noise.HomoscedasticNoise` the same block is
``p(log A) proportional to A^2`` — half-log-determinant exactly linear in
``log A`` with slope ``+2.000000`` over six decades, improper upward. Same
prior, same model, different noise declaration; on the same power law with a
fixed floor added, ``d/d beta`` of the two comes out with opposite signs. That
is why the prior carries no noise model of its own.

The other half of the file is the refusals, and one of them is the reason the
determinant is taken by ``eigh``. On a block that is degenerate by construction
— an amplitude ``exp(a + b)``, the same parameter twice — ``slogdet`` returns
sign ``+1.0`` and ``+6.420496``, and ``cholesky`` succeeds with a positive
smallest pivot. Both come back plausible for a density that does not exist.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.inference import (
    Bind,
    Block,
    JeffreysPrior,
    Latent,
    ParameterSpace,
    SamplingPlan,
)
from rheplicant.inference.identifiability import DEFAULT_RANK_RTOL
from rheplicant.inference.noise import HomoscedasticNoise, RadiometerNoise
from rheplicant.inference.numpyro_bridge import to_numpyro_model
from rheplicant.inference.uncertainty import fisher_information

pytest.importorskip("numpyro")
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer.util import log_density as numpyro_log_density  # noqa: E402

N_TIME, N_FREQ = 8, 8
NU0 = 70e6

#: The flat value, measured. Every one of the nine grid points below returns it.
RADIOMETER_FLAT_HALF_LOGDET = 15.80169853

#: ``d(half-logdet)/d log A`` under HomoscedasticNoise. ``p(log A) ~ A^2``.
HOMOSCEDASTIC_LOG_AMP_SLOPE = 2.000000


@pytest.fixture(scope="module", autouse=True)
def _float64():
    """A determinant at 1e-17 of its largest eigenvalue is not a float32 claim."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


# --------------------------------------------------------------- test doubles --


class PowerLaw(AbstractOperator):
    """``mu = A (nu/nu0)^-beta + floor``, broadcast over time.

    ``floor`` is a fixed constant and not a latent. At ``floor = 0`` the
    radiometer variance ``sigma = |mu| f`` factorises exactly and the Jeffreys
    prior is flat; a non-zero floor is what breaks that, and both cases are
    measured below.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    log_amp: jax.Array
    beta: jax.Array
    floor: jax.Array

    def __call__(self, state):
        nu = state.coords.freq / NU0
        row = jnp.exp(self.log_amp) * nu ** (-self.beta) + self.floor
        return state.with_data(jnp.broadcast_to(row, (state.coords.time.size, nu.size)))


class DoubledAmplitude(AbstractOperator):
    """``mu = exp(a + b) (nu/nu0)^-beta`` — ``a`` and ``b`` are one parameter twice."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    a: jax.Array
    b: jax.Array
    beta: jax.Array

    def __call__(self, state):
        nu = state.coords.freq / NU0
        row = jnp.exp(self.a + self.b) * nu ** (-self.beta)
        return state.with_data(jnp.broadcast_to(row, (state.coords.time.size, nu.size)))


# -------------------------------------------------------------------- fixtures --


def make_state(n_time: int = N_TIME, n_freq: int = N_FREQ) -> State:
    return State(
        coords=Coordinates(
            time=jnp.arange(float(n_time)) * 2.0,
            freq=jnp.linspace(60e6, 85e6, n_freq),
        ),
        meta={"telescope": "RHINO", "obs_id": "jeffreys-001"},
    )


RADIOMETER = RadiometerNoise(channel_width=1e6, integration_time=1.0)
HOMOSCEDASTIC = HomoscedasticNoise(jnp.array(0.5))


def power_law_space(joint_prior=None, priors=None) -> ParameterSpace:
    """``(fg_log_amp, fg_beta)`` bound straight into the power law."""
    priors = priors or {}
    return ParameterSpace(
        latents=[
            Latent("fg_log_amp", init=jnp.array(7.8), prior=priors.get("fg_log_amp")),
            Latent("fg_beta", init=jnp.array(2.55), prior=priors.get("fg_beta")),
        ],
        bindings=[
            Bind("fg_log_amp", into=lambda p: p.log_amp),
            Bind("fg_beta", into=lambda p: p.beta),
        ],
        joint_prior=joint_prior,
    )


def power_law(floor: float = 0.0) -> PowerLaw:
    return PowerLaw(
        log_amp=jnp.array(7.8), beta=jnp.array(2.55), floor=jnp.array(floor)
    )


def forward_of(space: ParameterSpace, pipeline, state: State):
    def forward(values):
        return space.bind(pipeline, values)(state).data

    return forward


def doubled_space(joint_prior=None) -> ParameterSpace:
    return ParameterSpace(
        latents=[
            Latent("a", init=jnp.array(0.0)),
            Latent("b", init=jnp.array(0.0)),
            Latent("fg_beta", init=jnp.array(2.55)),
        ],
        bindings=[
            Bind("a", into=lambda p: p.a),
            Bind("b", into=lambda p: p.b),
            Bind("fg_beta", into=lambda p: p.beta),
        ],
        joint_prior=joint_prior,
    )


def doubled_pipeline() -> DoubledAmplitude:
    return DoubledAmplitude(a=jnp.array(0.0), b=jnp.array(0.0), beta=jnp.array(2.55))


# ----------------------------------------------------- it is off by default --


def test_a_space_declares_no_joint_prior_by_default():
    """The whole feature is opt-in: nothing changes for a space that ignores it."""
    assert power_law_space().joint_prior is None


# ------------------------------------------------------ the measured headline --


GRID = [(la, be) for la in (6.8, 7.8, 8.8) for be in (2.05, 2.55, 3.05)]


@pytest.mark.parametrize(("log_amp", "beta"), GRID)
def test_the_radiometer_jeffreys_prior_of_a_bare_power_law_is_exactly_flat(log_amp, beta):
    """Under the package DEFAULT noise model this prior is the flat prior.

    ``sigma = |mu| f`` gives ``N^-1 = 1/(mu^2 f^2)`` and ``J_{k,i} = mu_k
    g_i(nu_k)``, so every ``mu`` cancels: ``I_ij = (1 + 2 f^2)/f^2 sum_k g_i
    g_j``, a constant matrix. Two decades of amplitude and a unit of spectral
    index move it by nothing.
    """
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    space = power_law_space(joint_prior=prior)
    forward = forward_of(space, power_law(), make_state())
    value = prior.log_density(
        forward,
        {"fg_log_amp": jnp.array(log_amp), "fg_beta": jnp.array(beta)},
        RADIOMETER,
    )
    assert float(value) == pytest.approx(RADIOMETER_FLAT_HALF_LOGDET, abs=1e-8)


def test_the_radiometer_jeffreys_prior_has_a_zero_gradient():
    """Flat to the last digit is a claim about the value; this is the derivative.

    Measured ``(0.000000e+00, 4.163336e-17)`` — the second entry is one ulp of
    the value itself, which is the strongest statement floating point can make.
    """
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    space = power_law_space(joint_prior=prior)
    forward = forward_of(space, power_law(), make_state())

    def logp(values):
        return prior.log_density(forward, values, RADIOMETER)

    grad = jax.grad(logp)(
        {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)}
    )
    assert abs(float(grad["fg_log_amp"])) < 1e-14
    assert abs(float(grad["fg_beta"])) < 1e-14


def test_under_homoscedastic_noise_the_amplitude_prior_is_A_squared():
    """``p(log A) ~ A^2``: slope ``+2.000000`` in ``log A``, over six decades.

    Improper upward — the density grows without bound with the amplitude — so
    this is emphatically not a neutral default, and it is a different prior
    from the one the identical declaration gives under RadiometerNoise.
    """
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    space = power_law_space(joint_prior=prior)
    forward = forward_of(space, power_law(), make_state())

    def logp(log_amp):
        return prior.log_density(
            forward,
            {"fg_log_amp": jnp.array(log_amp), "fg_beta": jnp.array(2.55)},
            HOMOSCEDASTIC,
        )

    low, high = float(logp(-3.0)), float(logp(3.0))
    assert (high - low) / 6.0 == pytest.approx(HOMOSCEDASTIC_LOG_AMP_SLOPE, abs=5e-7)
    # Linear, not merely with that average slope: the midpoint is on the line.
    assert float(logp(0.0)) == pytest.approx(0.5 * (low + high), abs=5e-7)


def test_the_noise_model_chooses_the_priors_shape():
    """One model, one block, two noise declarations, opposite signs in beta.

    On the power law with a fixed 300 K floor — where the radiometer variance
    no longer factorises and the prior stops being flat — ``d/d beta`` is
    ``-1.366854e-02`` under RadiometerNoise and ``+8.052944e-03`` under
    HomoscedasticNoise. The prior carries no noise model of its own precisely
    so that this choice cannot be made twice, differently, in one run.
    """
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    space = power_law_space(joint_prior=prior)
    forward = forward_of(space, power_law(floor=300.0), make_state())
    at = {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)}

    def slope(noise):
        return float(
            jax.grad(lambda v: prior.log_density(forward, v, noise))(at)["fg_beta"]
        )

    assert slope(RADIOMETER) == pytest.approx(-1.366854e-02, rel=1e-5)
    assert slope(HOMOSCEDASTIC) == pytest.approx(+8.052944e-03, rel=1e-5)
    assert slope(RADIOMETER) * slope(HOMOSCEDASTIC) < 0.0


# ------------------------------------------------ reparameterisation invariance --


def u_space(joint_prior=None) -> ParameterSpace:
    """The same model in ``u``, with ``beta = 2 + exp(u)`` — monotone, smooth."""
    return ParameterSpace(
        latents=[
            Latent("fg_log_amp", init=jnp.array(7.8)),
            Latent("u", init=jnp.log(jnp.array(0.55))),
        ],
        bindings=[
            Bind("fg_log_amp", into=lambda p: p.log_amp),
            Bind("u", into=lambda p: p.beta, fn=lambda u: 2.0 + jnp.exp(u)),
        ],
        joint_prior=joint_prior,
    )


@pytest.mark.parametrize("noise", [RADIOMETER, HOMOSCEDASTIC], ids=["radiometer", "homo"])
@pytest.mark.parametrize("beta", [2.20, 2.55, 3.30])
def test_it_is_reparameterisation_invariant(noise, beta):
    """The defining property, and the reason to want this prior at all.

    ``p(u) = p(beta) |d beta/d u|``, so in log terms the ``u``-space density
    must equal the ``beta``-space one plus ``log|d beta/d u|`` — exactly, not
    approximately. Measured agreement here is ~1e-15 absolute on numbers of
    order 15, i.e. at the last bit. A prior that failed this would be a
    different prior in every coordinate system a user might reach for, which is
    exactly what ``Normal(2.3, 0.3)`` on ``beta`` is and what this is not.
    """
    pipeline = power_law(floor=300.0)
    state = make_state()
    prior_beta = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    prior_u = JeffreysPrior(over=("fg_log_amp", "u"))

    in_beta = prior_beta.log_density(
        forward_of(power_law_space(), pipeline, state),
        {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(beta)},
        noise,
    )
    u = jnp.log(jnp.array(beta - 2.0))
    in_u = prior_u.log_density(
        forward_of(u_space(), pipeline, state),
        {"fg_log_amp": jnp.array(7.8), "u": u},
        noise,
    )
    log_jacobian = float(u)  # d beta / d u = exp(u) = beta - 2
    assert float(in_u) == pytest.approx(float(in_beta) + log_jacobian, abs=1e-11)


# ------------------------------------------------------ the singular block --


def singular_information() -> jax.Array:
    """The information matrix of the doubled-amplitude block, at its init."""
    space = doubled_space()
    forward = forward_of(space, doubled_pipeline(), make_state())
    prior = JeffreysPrior(over=("a", "b", "fg_beta"))
    return prior.information(
        forward,
        {"a": jnp.array(0.0), "b": jnp.array(0.0), "fg_beta": jnp.array(2.55)},
        RADIOMETER,
    )


def test_slogdet_and_cholesky_both_return_plausible_numbers_on_the_singular_block():
    """The reason the determinant is not taken either obvious way.

    ``a`` and ``b`` enter only as ``a + b``, so this matrix is exactly rank 2 of
    3 and its determinant is exactly zero. Neither routine says so: the null
    eigenvalue lands at ``-2.117e-09`` against a largest of ``1.281e+08``, i.e.
    1.7e-17 relative, and the SIGN of that roundoff is what decides whether
    ``slogdet`` reports ``-inf`` or a finite number and whether ``cholesky``
    returns NaN or a factor.

    Here it lands positive and both succeed. If this assertion ever fails on
    another platform's BLAS, the roundoff sign flipped and the claim being made
    is unchanged: neither routine RAISES, so neither is a guard. The load-
    bearing pins are the two tests below it, which do not depend on that sign.
    """
    matrix = singular_information()
    sign, logabsdet = jnp.linalg.slogdet(matrix)
    assert float(sign) == pytest.approx(1.0), (
        "slogdet reported a non-positive sign; the roundoff sign of the null "
        "eigenvalue flipped on this platform. The claim under test is that "
        "slogdet does not RAISE on a singular matrix, which still holds."
    )
    # Centre = the arm64 measurement; the band admits a decade of roundoff
    # in the null eigenvalue (log(10)/2 = 1.15). Measured 6.420496 on arm64
    # macOS and 6.444212 on x86_64 Linux -- the null eigenvalue differing by
    # about 5 %, which `logabsdet` carries as the LOG of, so a factor of two
    # in it moves this by 0.35. The old `abs=5e-6` was five orders of
    # magnitude tighter than the quantity is reproducible to, and passed only
    # on the machine it was written on.
    assert jnp.isfinite(logabsdet), "slogdet returned -inf on a singular matrix"
    assert float(0.5 * logabsdet) == pytest.approx(6.43, abs=1.2)

    factor = jnp.linalg.cholesky(matrix)
    assert bool(jnp.all(jnp.isfinite(factor))), (
        "cholesky returned NaN; the roundoff sign flipped on this platform. "
        "It still did not raise."
    )
    pivots = jnp.diag(factor)
    # The smallest pivot is the square root of that same roundoff, so it moves
    # by half the relative swing. Positive and of this order is the property.
    assert float(jnp.min(pivots)) > 0.0, "a non-positive pivot means cholesky failed"
    assert float(jnp.min(pivots)) == pytest.approx(9.755e-05, rel=0.5)
    # The SAME quantity as `0.5 * logabsdet` above, by log det = 2 sum log
    # pivot, down a different route -- so the two need not agree to the digit
    # and measurably do not: 6.420496 vs 6.566517 on arm64, while on x86_64
    # both land on 6.444212. That the gap BETWEEN the routes is itself
    # platform-dependent is the clearest statement that neither is a contract.
    assert float(jnp.sum(jnp.log(pivots))) == pytest.approx(6.43, abs=1.2)


def test_the_eigh_route_floors_the_singular_block_to_effectively_zero():
    """What this prior returns where the other two returned ``+6.42``.

    Not ``-inf``: an infinite potential is a NaN gradient rather than a rejected
    proposal. The floor is the smallest positive float64, so the answer is
    about ``-337`` — a density of ``e^-337``, which is zero for every purpose a
    sampler has.
    """
    prior = JeffreysPrior(over=("a", "b", "fg_beta"))
    value = float(prior.half_log_determinant(singular_information()))
    assert jnp.isfinite(value)
    assert value < -300.0
    assert value == pytest.approx(-338.05, abs=0.05)


def test_the_rank_floor_is_identifiabilitys_own_tolerance_by_default():
    """One cut, justified in one place, read from there rather than restated."""
    assert JeffreysPrior(over="a").rank_tolerance == DEFAULT_RANK_RTOL
    assert JeffreysPrior(over="a", rank_rtol=1e-6).rank_tolerance == 1e-6


def test_a_well_conditioned_block_agrees_with_slogdet():
    """The floor must not be doing anything where there is nothing to floor.

    On the non-degenerate power-law block the eigh route and ``slogdet`` agree
    to 1e-12; the disagreement above is the singular matrix, not the method.
    """
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    space = power_law_space()
    forward = forward_of(space, power_law(floor=300.0), make_state())
    matrix = prior.information(
        forward, {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)}, RADIOMETER
    )
    sign, logabsdet = jnp.linalg.slogdet(matrix)
    assert float(sign) == 1.0
    assert float(prior.half_log_determinant(matrix)) == pytest.approx(
        float(0.5 * logabsdet), abs=1e-9
    )


# ----------------------------------------------------------------- refusals --


def test_over_is_mandatory_and_a_bare_string_is_one_name():
    with pytest.raises(TypeError):
        JeffreysPrior()
    assert JeffreysPrior(over="fg_beta").over == ("fg_beta",)


def test_an_empty_block_is_refused():
    with pytest.raises(ParameterSpaceError, match="over no latents"):
        JeffreysPrior(over=())


def test_a_repeated_latent_in_over_is_refused():
    with pytest.raises(ParameterSpaceError, match="more than once"):
        JeffreysPrior(over=("a", "a"))


def test_a_non_string_name_in_over_is_refused():
    with pytest.raises(ParameterSpaceError, match="takes latent NAMES"):
        JeffreysPrior(over=(1, 2))


def test_a_non_positive_rank_tolerance_is_refused():
    with pytest.raises(ParameterSpaceError, match="positive relative cut"):
        JeffreysPrior(over="a", rank_rtol=0.0)


def test_over_naming_a_latent_the_space_does_not_have_is_refused():
    """Refusal 1. The block would silently shrink to the names that matched."""
    with pytest.raises(ParameterSpaceError, match=r"names \['fg_index'\]"):
        power_law_space(joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_index")))


def test_a_latent_carrying_both_priors_is_refused():
    """Refusal 2. Two priors on one quantity, multiplied, with no symptom."""
    with pytest.raises(ParameterSpaceError, match="AND declare their own"):
        power_law_space(
            joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta")),
            priors={"fg_beta": dist.Normal(2.3, 0.3)},
        )


def test_a_latent_outside_the_block_may_still_declare_its_own_prior():
    """The refusal is about DOUBLE priors, not about mixing the two kinds."""
    space = ParameterSpace(
        latents=[
            Latent("fg_log_amp", init=jnp.array(7.8)),
            Latent("fg_beta", init=jnp.array(2.55)),
            Latent("t_floor", init=jnp.array(300.0), prior=dist.Normal(300.0, 10.0)),
        ],
        bindings=[
            Bind("fg_log_amp", into=lambda p: p.log_amp),
            Bind("fg_beta", into=lambda p: p.beta),
            Bind("t_floor", into=lambda p: p.floor),
        ],
        joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta")),
    )
    assert space.joint_prior.over == ("fg_log_amp", "fg_beta")


def test_a_rank_deficient_block_is_refused_by_name():
    """Refusal 3, delegated to identifiability so the direction is named."""
    prior = JeffreysPrior(over=("a", "b", "fg_beta"))
    space = doubled_space(joint_prior=prior)
    with pytest.raises(ParameterSpaceError) as excinfo:
        prior.check_identified(space, doubled_pipeline(), make_state())
    message = str(excinfo.value)
    assert "nullity 1 of 3" in message
    assert "sqrt(det I) is not a density" in message
    assert "slogdet" in message and "cholesky" in message
    # The direction is named as a combination of latents, 0.50/0.50.
    assert "a 0.50" in message and "b 0.50" in message


def test_an_identified_block_passes_the_check_and_returns_its_report():
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    space = power_law_space(joint_prior=prior)
    report = prior.check_identified(space, power_law(), make_state())
    assert (report.rank, report.nullity) == (2, 0)


def test_evaluating_at_a_values_dict_missing_a_block_member_is_refused():
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    forward = forward_of(power_law_space(), power_law(), make_state())
    with pytest.raises(ParameterSpaceError, match=r"no entry for \['fg_beta'\]"):
        prior.log_density(forward, {"fg_log_amp": jnp.array(7.8)}, RADIOMETER)


def test_fisher_information_with_a_space_is_refused_while_one_is_declared():
    """Refusal 5: the prior is defined FROM this matrix; it cannot be inside it."""
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    space = power_law_space(joint_prior=prior)
    forward = forward_of(space, power_law(), make_state())
    values = {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)}
    with pytest.raises(ParameterSpaceError, match="inside its own definition"):
        fisher_information(forward, values, RADIOMETER, space=space)
    # space=None is the quantity the prior itself is built from, and still works.
    assert fisher_information(forward, values, RADIOMETER).kind == "fisher"


def test_a_block_straddling_two_plan_blocks_says_so_as_well():
    """Refusal 4. A plan never reads the prior, and a split one could not.

    Both partitions are refused, but the message distinguishes them: half a
    joint prior is not a conditional of anything, which is worth saying on top
    of the general "no plan evaluates this".
    """
    space = power_law_space(joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta")))
    with pytest.raises(ParameterSpaceError, match="splits it across blocks"):
        SamplingPlan(space, Block("fg_log_amp"), Block("fg_beta"))


def test_even_an_unsplit_block_is_refused_because_no_plan_reads_the_prior():
    """The advice "put the whole block in ONE Block" led into the silent case.

    ``engines._log_prior`` builds a block's conditional from ``Latent.prior``
    alone, and a covered latent declares none -- so the joint prior contributes
    exactly zero. Measured on this very partition: the conditional potential is
    identical with the declaration and without it, at every point, while
    ``0.5 logdet I`` ranges over 1.20 nats across the same points. The plan then
    runs and reports a converged chi-squared computed from blocks that never saw
    the prior, which is the failure this package exists to refuse -- arriving
    through the branch that USED to pass.
    """
    space = power_law_space(joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta")))
    with pytest.raises(ParameterSpaceError, match="does not evaluate a joint prior"):
        SamplingPlan(space, Block("fg_log_amp", "fg_beta"))


# ------------------------------------------------------------- the NUTS exit --


def observed_data() -> jax.Array:
    state = make_state()
    return power_law()(state).data


def build_model(space, noise=RADIOMETER, **kwargs):
    return to_numpyro_model(power_law(), make_state(), space, noise, **kwargs)


def test_a_latent_the_joint_prior_covers_needs_no_latent_prior():
    """Without the exemption the bridge's own prior check refuses the space."""
    space = power_law_space(joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta")))
    model = build_model(space)
    data = observed_data()
    value, _ = numpyro_log_density(
        model,
        (),
        {"observed": data},
        {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)},
    )
    assert jnp.isfinite(value)


def test_a_latent_outside_the_block_still_needs_one():
    space = ParameterSpace(
        latents=[
            Latent("fg_log_amp", init=jnp.array(7.8)),
            Latent("fg_beta", init=jnp.array(2.55)),
            Latent("t_floor", init=jnp.array(300.0)),
        ],
        bindings=[
            Bind("fg_log_amp", into=lambda p: p.log_amp),
            Bind("fg_beta", into=lambda p: p.beta),
            Bind("t_floor", into=lambda p: p.floor),
        ],
        joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta")),
    )
    with pytest.raises(ParameterSpaceError, match=r"\['t_floor'\] have no prior"):
        to_numpyro_model(power_law(), make_state(), space, RADIOMETER)


def test_the_model_adds_exactly_the_half_log_determinant():
    """The factor site is the prior, not a rescaling of it.

    Differenced against the same space sampled flat, so the likelihood cancels
    and what is left is the prior term alone.
    """
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    with_prior = build_model(power_law_space(joint_prior=prior), noise=HOMOSCEDASTIC)
    flat = build_model(
        power_law_space(
            priors={
                "fg_log_amp": dist.ImproperUniform(dist.constraints.real, (), ()),
                "fg_beta": dist.ImproperUniform(dist.constraints.real, (), ()),
            }
        ),
        noise=HOMOSCEDASTIC,
    )
    data = observed_data()
    forward = forward_of(power_law_space(), power_law(), make_state())
    for log_amp, beta in ((7.8, 2.55), (8.3, 2.20)):
        params = {"fg_log_amp": jnp.array(log_amp), "fg_beta": jnp.array(beta)}
        a, _ = numpyro_log_density(with_prior, (), {"observed": data}, params)
        b, _ = numpyro_log_density(flat, (), {"observed": data}, params)
        expected = float(prior.log_density(forward, params, HOMOSCEDASTIC))
        assert float(a - b) == pytest.approx(expected, abs=1e-6)


def test_under_radiometer_noise_switching_it_on_only_shifts_the_posterior():
    """The headline, at the exit: on this model it is the flat prior.

    The difference between the two log-posteriors is the SAME constant at two
    very different parameter points, which is what "this changes nothing about
    the shape" means operationally.
    """
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    with_prior = build_model(power_law_space(joint_prior=prior))
    flat = build_model(
        power_law_space(
            priors={
                "fg_log_amp": dist.ImproperUniform(dist.constraints.real, (), ()),
                "fg_beta": dist.ImproperUniform(dist.constraints.real, (), ()),
            }
        )
    )
    data = observed_data()
    shifts = []
    for log_amp, beta in ((7.8, 2.55), (8.6, 2.05), (6.9, 3.05)):
        params = {"fg_log_amp": jnp.array(log_amp), "fg_beta": jnp.array(beta)}
        a, _ = numpyro_log_density(with_prior, (), {"observed": data}, params)
        b, _ = numpyro_log_density(flat, (), {"observed": data}, params)
        shifts.append(float(a - b))
    assert shifts[0] == pytest.approx(RADIOMETER_FLAT_HALF_LOGDET, abs=1e-6)
    # 1e-7 and not tighter because these are DIFFERENCES of log-posteriors of
    # order 1e5: the constant is exact, the subtraction is not. The parametrized
    # test above pins the prior itself to 1e-8 with no cancellation in it.
    assert shifts[1] == pytest.approx(shifts[0], abs=1e-7)
    assert shifts[2] == pytest.approx(shifts[0], abs=1e-7)


def test_the_bridge_refuses_a_rank_deficient_block_before_sampling():
    """The build-time half of refusal 3, on the exit that would have sampled it."""
    space = doubled_space(joint_prior=JeffreysPrior(over=("a", "b", "fg_beta")))
    with pytest.raises(ParameterSpaceError, match="sqrt.det I. is not a density"):
        to_numpyro_model(doubled_pipeline(), make_state(), space, RADIOMETER)


def test_a_sampled_noise_std_with_a_jeffreys_prior_is_refused():
    """Refusal 6, the biggest measured hazard.

    ``numpyro_bridge`` creates a sample site ``"noise_std"`` when ``noise_std``
    is a distribution. That site is in NO ParameterSpace, so no ``over=`` can
    name it — and with a Jeffreys prior active the sigma posterior is silently
    multiplied by ``sigma^-p``, ``p = len(over)``. The exponent is measured in
    :func:`test_a_jeffreys_prior_tilts_a_sampled_sigma_by_exactly_minus_p`; what
    it costs follows from it in closed form, and is 1.0 sigma at ``p = 32``.
    """
    space = power_law_space(joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta")))
    with pytest.raises(ParameterSpaceError, match=r"sigma\^-2"):
        to_numpyro_model(
            power_law(), make_state(), space, dist.HalfNormal(1.0)
        )


@pytest.mark.parametrize("over", [("fg_log_amp",), ("fg_log_amp", "fg_beta")])
def test_a_jeffreys_prior_tilts_a_sampled_sigma_by_exactly_minus_p(over):
    """The mechanism behind refusal 6, measured rather than asserted.

    Under HomoscedasticNoise ``I = J^T J / sigma^2`` over a ``p``-latent block,
    so ``0.5 log det I = -p log sigma + const`` and the factor site is a
    ``sigma^-p`` multiplier on a sigma the user may well have thought had only
    the prior they declared for it. The derivative is exact, not asymptotic:
    ``-1.000000000`` and ``-2.000000000`` here.

    The block stops at two on this fixture for a reason worth recording. Adding
    ``t_floor`` makes it numerically rank-deficient — over 60-85 MHz a power law
    is nearly a straight line, so amplitude and index between them nearly span a
    constant, and the third eigenvalue lands at ``0.0459`` against a largest of
    ``2.139e+08``, i.e. 2.1e-10 relative and below the rank floor. The measured
    slope there is ``-2`` rather than ``-3``, because the floored eigenvalue
    contributes a constant instead of another ``log sigma``. That is the floor
    reporting a real near-degeneracy, and ``check_identified`` refuses the same
    block up front.

    From there the cost is arithmetic. The sigma posterior's mode moves from
    ``chi2/n`` to ``chi2/(n + p)``, which against a ``log sigma`` width of
    ``1/sqrt(2n)`` is ``p/sqrt(2n)`` standard deviations: 0.062 sigma at
    ``p = 2`` over 512 samples, and 1.0 sigma at ``p = 32``.
    """
    space = ParameterSpace(
        latents=[
            Latent("fg_log_amp", init=jnp.array(7.8)),
            Latent("fg_beta", init=jnp.array(2.55)),
            Latent("t_floor", init=jnp.array(300.0)),
        ],
        bindings=[
            Bind("fg_log_amp", into=lambda p: p.log_amp),
            Bind("fg_beta", into=lambda p: p.beta),
            Bind("t_floor", into=lambda p: p.floor),
        ],
    )
    prior = JeffreysPrior(over=over)
    forward = forward_of(space, power_law(floor=300.0), make_state())
    values = {
        "fg_log_amp": jnp.array(7.8),
        "fg_beta": jnp.array(2.55),
        "t_floor": jnp.array(300.0),
    }

    def logp(log_sigma):
        return prior.log_density(
            forward, values, HomoscedasticNoise(jnp.exp(log_sigma))
        )

    slope = float(jax.grad(logp)(jnp.array(0.3)))
    assert slope == pytest.approx(-float(len(over)), abs=1e-9)


def test_the_sampled_noise_std_refusal_can_be_opted_into():
    """A caller who has read the number may still want it; they say so."""
    space = power_law_space(joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta")))
    model = to_numpyro_model(
        power_law(),
        make_state(),
        space,
        dist.HalfNormal(1.0),
        allow_sampled_noise_std=True,
    )
    value, _ = numpyro_log_density(
        model,
        (),
        {"observed": observed_data()},
        {
            "fg_log_amp": jnp.array(7.8),
            "fg_beta": jnp.array(2.55),
            "noise_std": jnp.array(0.5),
        },
    )
    assert jnp.isfinite(value)


def test_the_opt_in_is_inert_without_a_joint_prior():
    """A sampled sigma alone was always fine and stays fine, unflagged."""
    space = power_law_space(
        priors={
            "fg_log_amp": dist.Normal(7.8, 1.0),
            "fg_beta": dist.Normal(2.55, 0.3),
        }
    )
    model = to_numpyro_model(power_law(), make_state(), space, dist.HalfNormal(1.0))
    value, _ = numpyro_log_density(
        model,
        (),
        {"observed": observed_data()},
        {
            "fg_log_amp": jnp.array(7.8),
            "fg_beta": jnp.array(2.55),
            "noise_std": jnp.array(0.5),
        },
    )
    assert jnp.isfinite(value)


def test_the_prior_inherits_the_exits_noise_rather_than_carrying_one():
    """A likelihood/prior noise mismatch is not expressible through this API.

    The same space, built at the same point, under two noise models: the factor
    site follows the noise the EXIT was given, with nothing on the prior to set
    separately.
    """
    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    space = power_law_space(joint_prior=prior)
    forward = forward_of(power_law_space(), power_law(), make_state())
    params = {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)}
    data = observed_data()

    flat_space = power_law_space(
        priors={
            "fg_log_amp": dist.ImproperUniform(dist.constraints.real, (), ()),
            "fg_beta": dist.ImproperUniform(dist.constraints.real, (), ()),
        }
    )
    for noise in (RADIOMETER, HOMOSCEDASTIC):
        a, _ = numpyro_log_density(
            build_model(space, noise=noise), (), {"observed": data}, params
        )
        b, _ = numpyro_log_density(
            build_model(flat_space, noise=noise), (), {"observed": data}, params
        )
        assert float(a - b) == pytest.approx(
            float(prior.log_density(forward, params, noise)), abs=1e-6
        )


def test_information_rows_are_in_sorted_order_not_declaration_order():
    """A public matrix whose row order is not the order you asked for.

    ``fisher_information`` flattens by sorted key, so ``over`` is only a set as
    far as the returned matrix is concerned. It does not affect the prior --
    ``det`` is invariant under a symmetric permutation, which the second
    assertion pins -- but a caller reading rows positionally against ``over``
    gets the wrong latent, and nothing in the signature says so.
    """
    forward_a = forward_of(
        power_law_space(joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta"))),
        power_law(),
        make_state(),
    )
    values = {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)}
    declared = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    reversed_ = JeffreysPrior(over=("fg_beta", "fg_log_amp"))

    forward_matrix = declared.information(forward_a, values, RADIOMETER)
    reverse_matrix = reversed_.information(forward_a, values, RADIOMETER)

    assert jnp.allclose(forward_matrix, reverse_matrix), (
        "Reversing over= changed the matrix, so the row order now follows the "
        "declaration. Update the docstring's Returns paragraph with it."
    )
    # sorted(("fg_log_amp", "fg_beta")) == ("fg_beta", "fg_log_amp"), so row 0
    # is fg_beta whichever way the block was declared.
    assert float(forward_matrix[0, 0]) == pytest.approx(
        float(reverse_matrix[0, 0]), rel=1e-15
    )
    assert float(declared.log_density(forward_a, values, RADIOMETER)) == pytest.approx(
        float(reversed_.log_density(forward_a, values, RADIOMETER)), rel=1e-15
    )


# ----------------------------------------------------- and now an actual chain --


def _chain(space, noise, key, data):
    """One NUTS run, from the declared start, on the given key."""
    from numpyro.infer import MCMC, NUTS

    from rheplicant.inference.numpyro_bridge import init_to_declared

    kernel = NUTS(
        build_model(space, noise=noise), init_strategy=init_to_declared(space)
    )
    mcmc = MCMC(kernel, num_warmup=200, num_samples=300, progress_bar=False)
    mcmc.run(key, observed=data)
    return mcmc.get_samples()


def _flat_priors():
    return {
        name: dist.ImproperUniform(dist.constraints.real, (), ())
        for name in ("fg_log_amp", "fg_beta")
    }


class TestASamplerActuallyCarriesTheFactor:
    """The acceptance the potential tests above are silent about.

    Everything before this evaluates the potential at points chosen by hand.
    That is the right instrument for "is the factor the number it should be",
    and it cannot say the two things only a run can: that NUTS differentiates
    through the factor site at every leapfrog step without producing a NaN, and
    that the site is in the potential the sampler explores rather than merely
    in a ``log_density`` someone can call.

    **What a chain cannot do here is measure the prior's displacement.**
    ``prior_sensitivity``'s own module docstring states the arithmetic: the
    Monte Carlo standard error of a posterior mean from ``n_eff`` draws is
    ``1/sqrt(n_eff)`` sigma, and two chains' noise ADDS. Measured under a
    homoscedastic declaration where this prior is ``p(log A) ~ A^2``: the mean
    shift is +0.005 against a posterior sd of 0.055, and across three seeds the
    first-order prediction was matched at ratios 0.55, 1.44 and 1.09 -- with
    the beta component changing SIGN.

    **And a chain cannot compare TRAJECTORIES either, which cost this file two
    wrong assertions and five red CI runs.** A constant added to a potential
    leaves a NUTS trajectory unchanged in exact arithmetic, so the first
    version asserted that two chains -- one carrying a flat Jeffreys factor,
    one not -- agree bitwise under a common key. They did, on the machine it
    was written on. They do not on another: the radiometer factor's every-``mu``
    cancellation is exact only to roundoff, a leapfrog trajectory is chaotic, and
    the last bits grow. Re-pinning it to "1e-5 of a posterior sd" was the same
    mistake with a tolerance on it -- measured 1.4e-6 locally, **1.89e-2** on
    the CI runner, four orders out.

    So the flat half is NOT tested here. It is tested where it is deterministic
    --- ``test_under_radiometer_noise_switching_it_on_only_shifts_the_posterior``
    differences the two potentials at three widely separated points and pins the
    constant to 1e-7. What is left for a chain is what only a chain can say, and
    both remaining assertions are about EFFECTS THAT ARE LARGE: a chain that
    runs and moves, and a non-flat prior that visibly relocates it.
    """

    def test_a_flat_prior_still_gives_a_healthy_chain(self):
        """NUTS differentiates through the factor at every step.

        No cross-chain comparison: the claim is that the sampler can carry this
        factor at all, which a NaN potential or a stuck chain would break and
        which no machine's rounding can fake.
        """
        data = observed_data()
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        samples = _chain(
            power_law_space(joint_prior=prior), RADIOMETER,
            jax.random.key(20260827), data,
        )
        for name in ("fg_log_amp", "fg_beta"):
            assert jnp.all(jnp.isfinite(samples[name])), f"{name} went non-finite"
            assert float(jnp.std(samples[name])) > 0.0, (
                f"{name} never moved: the chain is stuck, which is what an "
                "infinite or NaN factor looks like from here"
            )

    def test_a_prior_that_is_not_flat_moves_the_trajectory(self):
        """The half a chain CAN state, because the effect is large.

        Under a constant sigma this prior is ``p(log A) ~ A^2`` and is not
        constant, so two chains on one key must separate. Measured at four to
        five posterior standard deviations -- an effect no machine's rounding
        reaches, which is the whole difference between this assertion and the
        two that were removed.
        """
        data = observed_data()
        key = jax.random.key(20260827)
        noise = HomoscedasticNoise(sigma=jnp.array(1000.0))
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        with_prior = _chain(power_law_space(joint_prior=prior), noise, key, data)
        without = _chain(power_law_space(priors=_flat_priors()), noise, key, data)
        for name in ("fg_log_amp", "fg_beta"):
            spread = float(jnp.std(without[name]))
            difference = float(jnp.max(jnp.abs(with_prior[name] - without[name])))
            assert difference > spread, (
                f"{name}: the chains differ by {difference:.3e} against a "
                f"posterior sd of {spread:.3e}. Identical trajectories mean the "
                "factor never reached the potential the sampler explores."
            )


def test_a_vector_latent_permutes_by_ITS_SPAN_and_not_as_one_row():
    """D24's permutation is over spans, and a two-scalar block cannot say so.

    The rows come back from the far side in ``over``'s order and this package
    lays them out by sorted key, so the facade permutes. With every latent a
    scalar that is a permutation of names; with a vector in the block it is a
    permutation of SPANS, and a version that swapped names as single rows would
    return a matrix of the right shape, symmetric, positive definite, and
    scrambled.

    The oracle is ``fisher_information`` -- this package's own assembly, which
    flattens by sorted key and is what ``information`` called before it
    delegated. It is a REGRESSION oracle rather than an independent one, and it
    is available only until ``uncertainty`` is switched in its turn; when that
    happens this comparison retires with it and the layout claim needs a home
    that does not depend on the old spelling still being here.
    """
    from rheplicant.inference.uncertainty import fisher_information

    freq = jnp.linspace(60e6, 85e6, N_FREQ) / NU0

    def forward(values):
        row = jnp.exp(values["fg_log_amp"]) * freq ** (-values["fg_beta"])
        return jnp.broadcast_to(row, (N_TIME, N_FREQ)) + values["offsets"][:, None]

    values = {
        "fg_log_amp": jnp.array(7.8),
        "fg_beta": jnp.array(2.55),
        "offsets": jnp.array([3.0, -1.0, 2.0, 0.5, -2.0, 1.0, 0.0, 4.0]),
    }
    over = ("offsets", "fg_log_amp", "fg_beta")
    prior = JeffreysPrior(over=over)
    mine = prior.information(forward, values, HOMOSCEDASTIC)

    block = {name: values[name] for name in over}
    held = {k: v for k, v in values.items() if k not in block}
    expected = fisher_information(
        lambda moving: forward({**held, **moving}),
        block,
        HOMOSCEDASTIC,
        None,
        space=None,
    ).matrix
    expected = 0.5 * (expected + expected.T)

    assert mine.shape == (10, 10)
    assert jnp.allclose(mine, expected, rtol=1e-10, atol=0.0), (
        "the permuted block does not match this package's own sorted-key "
        f"layout; worst entry differs by {float(jnp.max(jnp.abs(mine - expected))):.3e}"
    )
    # sorted(over) is ("fg_beta", "fg_log_amp", "offsets"): two scalars, then
    # the eight-element span. A name-wise permutation would put `offsets` in a
    # single row and the two scalars in eight.
    assert float(mine[0, 0]) == pytest.approx(float(expected[0, 0]), rel=1e-10)
    assert float(mine[2, 2]) == pytest.approx(float(expected[2, 2]), rel=1e-10)


class TestTheSynthesisedInformationGraph:
    """The facade invents two things, and their legality is that the answer
    cannot reach them. That is a claim about the far side, so it is MEASURED.

    ``JeffreysPrior.information`` is handed ``f(values) -> prediction`` and a
    values dict. ``to_graph`` wants a space and a pipeline, so the adapter
    builds the three-layer graph from the callable instead
    (``graph_bridge.graph_for_information``) and has to supply what a graph
    requires and a Fisher block does not: **data**, and a **density per
    latent**. Both are chosen to be unreachable -- a Fisher information is an
    EXPECTED information, so no residual appears in it, and the block's prior
    fields are empty on the far side by that function's own docstring.

    **A docstring is not a measurement**, and this is the assumption the whole
    delegation rests on, so the graph is rebuilt with each synthesised thing
    changed and the matrices are compared. Same shape as D22's
    ``TestTheSynthesisedGraph`` for the rank test, and for the same reason.

    Three latents, not two: the covered ones MUST stay improper -- the far side
    refuses two priors on one quantity by type -- so varying a density needs a
    latent the block does not name. ``t_floor`` is that latent, held fixed, and
    it reaches the prediction so that its presence is not free.
    """

    OVER = ("fg_log_amp", "fg_beta")
    VALUES = {
        "fg_log_amp": jnp.array(7.8),
        "fg_beta": jnp.array(2.55),
        "t_floor": jnp.array(300.0),
    }

    @staticmethod
    def _graph(data, held_density):
        """The adapter's own construction, with the two knobs exposed."""
        import bayesmith

        from rheplicant.inference import graph_bridge as gb

        names = ("fg_log_amp", "fg_beta", "t_floor")
        freq = jnp.linspace(60e6, 85e6, N_FREQ) / NU0
        flat = dist.ImproperUniform(dist.constraints.real, (), ())

        def forward(values):
            row = (
                jnp.exp(values["fg_log_amp"]) * freq ** (-values["fg_beta"])
                + values["t_floor"]
            )
            return jnp.broadcast_to(row, (N_TIME, N_FREQ))

        def model(observed):
            refs = [
                bayesmith.sample(
                    name,
                    gb._prior_factory(held_density if name == "t_floor" else flat),
                )
                for name in names
            ]
            prediction = bayesmith.det(
                gb.PREDICTION, gb._prediction_fn(forward, names), *refs
            )
            bayesmith.observe(
                gb.OBSERVATION,
                gb._observation_fn(RADIOMETER),
                prediction,
                obs=observed,
                mask=gb._observed_mask(RADIOMETER),
                depends_on_prediction=bool(RADIOMETER.depends_on_prediction),
            )

        return bayesmith.trace(model, data)

    @classmethod
    def _matrix(cls, graph):
        import bayesmith

        return bayesmith.JeffreysPrior(over=cls.OVER).information(graph, cls.VALUES)

    @classmethod
    def _baseline(cls):
        flat = dist.ImproperUniform(dist.constraints.real, (), ())
        return cls._matrix(cls._graph(jnp.zeros((N_TIME, N_FREQ)), flat))

    def test_the_baseline_is_not_degenerate(self):
        """Without this every comparison below is satisfied by three copies of
        the zero matrix, which would agree perfectly and mean nothing."""
        baseline = self._baseline()
        assert baseline.shape == (2, 2)
        assert float(jnp.linalg.det(baseline)) > 0.0
        eigenvalues = jnp.linalg.eigvalsh(baseline)
        assert float(eigenvalues[0]) > 0.0
        assert float(eigenvalues[-1] / eigenvalues[0]) > 10.0, (
            "the two eigenvalues are within a decade of each other, so a matrix "
            "that had lost its structure could still look like this one"
        )

    def test_the_synthesised_data_cannot_reach_the_answer(self):
        """Zeros against 1e4 counts -- four decades, and the answer does not
        move by a bit. A Fisher information carries no residual.

        **A mutation of the adapter's chosen data therefore CANNOT be killed,
        and that is the point rather than a gap.** Changing
        ``graph_for_information``'s ``jnp.zeros`` to ``jnp.full(..., 1e4)``
        leaves this whole suite green, which is what this test asserts is true.
        A guard that went red there would be claiming something false about an
        expected information.

        What can still go wrong is the adapter building a DIFFERENT graph from
        the one measured here, so the last assertion closes that loop: the
        shipped helper's matrix is the baseline's.
        """
        flat = dist.ImproperUniform(dist.constraints.real, (), ())
        other = self._matrix(self._graph(jnp.full((N_TIME, N_FREQ), 1e4), flat))
        baseline = self._baseline()
        assert float(jnp.max(jnp.abs(other - baseline))) == 0.0

        from rheplicant.inference.graph_bridge import graph_for_information

        freq = jnp.linspace(60e6, 85e6, N_FREQ) / NU0

        def forward(values):
            row = (
                jnp.exp(values["fg_log_amp"]) * freq ** (-values["fg_beta"])
                + values["t_floor"]
            )
            return jnp.broadcast_to(row, (N_TIME, N_FREQ))

        shipped = self._matrix(
            graph_for_information(forward, self.VALUES, RADIOMETER)
        )
        assert float(jnp.max(jnp.abs(shipped - baseline))) == 0.0, (
            "the adapter builds a different graph from the one these "
            "measurements were made on, so they say nothing about it"
        )

    def test_the_synthesised_densities_cannot_reach_the_answer(self):
        """A proper Normal in place of the flat declaration, on the latent the
        block does not name -- which is the one the adapter also declares flat
        and could equally have declared otherwise."""
        other = self._matrix(
            self._graph(jnp.zeros((N_TIME, N_FREQ)), dist.Normal(0.0, 1e6))
        )
        assert float(jnp.max(jnp.abs(other - self._baseline()))) == 0.0
