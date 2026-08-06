"""Section 8: freezing N is what makes the stored statistics constants.

Under ``RadiometerNoise`` sigma tracks the prediction (D21), so ``N^-1`` is a
function of the very quantity being inferred and T1's ``(chi2_r, p, R)`` are
not theta-independent constants at all. Freezing ``N`` is what makes them
constants -- and D23 records what the frozen answer then *is*: generalized
least squares, not the maximum of the full Gaussian likelihood. Two different
estimators, whose sum is neither, which is why ``noise_frozen_at`` travels on
the term and ``BayesMemory.remember`` refuses to add two of them.

**Provenance is not detection.** Recording that the covariance was frozen says
nothing about what freezing cost, and the cost is what a campaign is exposed
to. So it is measured, at ``2 n_theta + 1`` probes spanning the declared
support, and refused above a declared tolerance.

Measured on this fixture, one epoch, ``n_S = 6``, the declared box being the
truth +/- 3 prior sigma per latent:

===================  ===============
probe                 |frozen - live|
===================  ===============
the anchor            2.84e-14 nats
``t21_depth`` +/-3s   0.757, 0.685
``running`` +/-3s     30.6, 32.3
``index`` +/-3s       4190.6, 4178.3
``amplitude`` +/-3s   4.219e7, 1.654e7
===================  ===============

The direction that dominates is ``amplitude``, and not because the science
lives there: sigma is *proportional* to the prediction, so the latent that
scales the prediction is the one a frozen sigma is most wrong about. The
science direction costs under a nat over the same box. A single scalar
"freezing is fine" would have hidden both facts.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compress import compress_reduced_basis
from rheplicant.inference.compressed import RawLikelihood
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.memory import BayesMemory
from rheplicant.inference.noise import HomoscedasticNoise
from rheplicant.inference.reduced_basis import build_reduced_basis
from tests.evidence import rhino_bank

#: The whitened bank's numerical rank on this fixture. Anything above it is
#: refused by `build_reduced_basis`, not merely wasteful.
N_BASIS = 6

#: Measured, on the epoch below, over the declared support. The worst probe is
#: `amplitude` at its lower edge.
WORST_NATS = 4.2187778e7
ANCHOR_NATS = 2.842e-14


def _basis():
    return build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(jax.random.key(60)),
        n_basis=N_BASIS,
        support=rhino_bank.support(),
    )


def test_the_fractional_noise_is_the_number_d21_uses():
    noise = rhino_bank.noise()
    assert noise.fractional == pytest.approx(3.7722e-4, rel=1e-4)
    assert noise.fractional**2 == pytest.approx(1.4230e-7, rel=1e-3)


def test_freezing_is_recorded_as_provenance_and_gates_the_sum():
    basis, data = _basis(), rhino_bank.observed(jax.random.key(61))
    frozen = compress_reduced_basis(
        basis,
        observed=data,
        noise=rhino_bank.noise(),
        epoch_id="e0",
        noise_frozen_at="reference",
    )
    assert frozen.estimator == ("full", "reference")
    other = compress_reduced_basis(
        basis,
        observed=data,
        noise=rhino_bank.noise(),
        epoch_id="e1",
        noise_frozen_at="gls",
    )
    memory = BayesMemory(Factorization(rhino_bank.space())).remember(frozen)
    with pytest.raises(StateValidationError, match="estimator"):
        memory.remember(other)


def test_the_frozen_residual_is_measured_and_stored():
    """A number, not an argument that it is small."""
    term = compress_reduced_basis(
        _basis(),
        observed=rhino_bank.observed(jax.random.key(62)),
        noise=rhino_bank.noise(),
        epoch_id="e0",
        noise_frozen_at="reference",
    )
    assert term.frozen_noise_residual > 0.0
    assert np.isfinite(term.frozen_noise_residual)
    assert term.frozen_noise_residual == pytest.approx(WORST_NATS, rel=1e-3)


def test_a_prediction_independent_noise_model_freezes_to_exactly_zero():
    """`depends_on_prediction` is a claim, and this is what it claims.

    The residual is computed the same way for every noise model rather than
    behind a branch on that flag, so the zero below is *arithmetic* -- the two
    sigma arrays are the same array -- and not a skipped code path. A branch
    would make this test pass with the whole measurement deleted.
    """
    term = compress_reduced_basis(
        _basis(),
        observed=rhino_bank.observed(jax.random.key(63)),
        noise=HomoscedasticNoise(jnp.asarray(0.1)),
        epoch_id="e0",
    )
    assert term.frozen_noise_residual == 0.0
    assert term.noise_frozen_at == "none"


def test_the_residual_is_the_freezing_cost_alone_not_the_projection_error():
    """Section 7 owns the projection error; this number must not double-count it.

    Under a constant sigma there is nothing to freeze, so the stored residual is
    exactly zero -- while T1 still differs from the raw likelihood by 8.2e-6
    nats over the same probes, because the prediction is expanded in six
    directions rather than reproduced. Measuring `T1 - live raw` instead would
    have reported that 8.2e-6 as a cost of freezing, and the refusal's message
    would have named the wrong remedy.
    """
    basis = _basis()
    data = rhino_bank.observed(jax.random.key(63))
    noise = HomoscedasticNoise(jnp.asarray(0.1))
    term = compress_reduced_basis(basis, observed=data, noise=noise, epoch_id="e0")
    values = dict(basis.reference_values)
    probes = [dict(values)]
    for name, (low, high) in basis.support.items():
        probes += [{**values, name: jnp.asarray(edge)} for edge in (low, high)]
    projection = max(
        abs(
            float(term(probe))
            - float(
                RawLikelihood(
                    predict=basis.predict,
                    observed=data,
                    sigma=jnp.broadcast_to(
                        noise.std(basis.predict(probe)), data.shape
                    ),
                    names=tuple(values),
                    epoch_id="e0",
                )(probe)
            )
        )
        for probe in probes
    )
    assert term.frozen_noise_residual == 0.0
    assert projection == pytest.approx(8.204e-6, rel=1e-2)


def test_compression_refuses_a_residual_above_the_declared_tolerance():
    with pytest.raises(StateValidationError, match="Freezing the covariance costs"):
        compress_reduced_basis(
            _basis(),
            observed=rhino_bank.observed(jax.random.key(64)),
            noise=rhino_bank.noise(),
            epoch_id="e0",
            noise_frozen_at="reference",
            frozen_tolerance=1e-18,
        )


def test_a_tolerance_the_epoch_actually_meets_does_not_fire():
    """The nearest legitimate case: a guard that over-refuses is its own bug."""
    term = compress_reduced_basis(
        _basis(),
        observed=rhino_bank.observed(jax.random.key(64)),
        noise=rhino_bank.noise(),
        epoch_id="e0",
        noise_frozen_at="reference",
        frozen_tolerance=1e9,
    )
    assert term.frozen_noise_residual < 1e9


def test_narrowing_the_support_is_the_remedy_the_message_names():
    """4.2e7 nats over the full box, 0.757 over the science axis alone.

    The refusal says "narrow the support", and this measures that it works --
    a remedy nobody checked is a sentence, not a remedy. It also says which
    axis the cost lives on: `amplitude` scales the prediction and sigma is
    proportional to the prediction, so the frozen covariance is wrong there by
    seven orders of magnitude more than it is along `t21_depth`.
    """
    data = rhino_bank.observed(jax.random.key(62))
    narrow = compress_reduced_basis(
        _basis(),
        observed=data,
        noise=rhino_bank.noise(),
        epoch_id="e0",
        noise_frozen_at="reference",
        support={"t21_depth": rhino_bank.support()["t21_depth"]},
        frozen_tolerance=1.0,
    )
    assert narrow.frozen_noise_residual == pytest.approx(0.7568, rel=1e-3)


def test_a_nan_residual_refuses_rather_than_losing_the_comparison():
    """NaN defeats `>` and would sail through; the guard is written `not <=`.

    A zero sigma makes the quadratic `-inf` and the log-determinant `+inf`, so
    the density is NaN -- and `nan > tolerance` is False, which is precisely
    the answer that lets a broken epoch past a comparison-based guard. The
    reduction over the probes is `np.max` on an array rather than Python's
    `max` on a list for the same reason: the builtin's answer depends on which
    element the NaN happened to be.
    """
    with pytest.raises(StateValidationError, match="Freezing the covariance costs"):
        compress_reduced_basis(
            _basis(),
            observed=rhino_bank.observed(jax.random.key(65)),
            noise=HomoscedasticNoise(jnp.asarray(0.0)),
            epoch_id="e0",
            frozen_tolerance=1e9,
        )


def test_the_measurement_does_not_cost_the_traces_this_module_pins():
    """`jax.grad` and `jax.vmap` over `observed` keep working.

    Measured: taking the residual as a Python float concretised the data and
    turned both of `test_reduced_basis_likelihood.py`'s trace pins into a
    ConcretizationTypeError. So the residual is a dynamic leaf, unlike
    `n_observed` beside it -- which is computable from sigma alone and is
    therefore static provenance.
    """
    basis = _basis()
    epoch = rhino_bank.observed(jax.random.key(66))
    probe = dict(basis.reference_values)

    def scalar(data):
        return compress_reduced_basis(
            basis, observed=data, noise=rhino_bank.noise(), epoch_id="e0"
        )(probe)

    assert bool(jnp.all(jnp.isfinite(jax.grad(scalar)(epoch))))
    stacked = jnp.stack([epoch, rhino_bank.observed(jax.random.key(67))])
    residuals = jax.vmap(
        lambda d: compress_reduced_basis(
            basis, observed=d, noise=rhino_bank.noise(), epoch_id="e0"
        ).frozen_noise_residual
    )(stacked)
    assert residuals.shape == (2,)
    assert bool(jnp.all(residuals > 0.0))


def test_a_tolerance_under_a_trace_is_refused_by_name():
    """The refusal cannot be enforced there, and says so rather than crashing.

    Left unguarded the symptom is a raw TracerBoolConversionError out of the
    comparison, which names neither the cause nor the remedy -- the same
    complaint `_refuse_traced_sigma` exists to answer one seam over.
    """
    basis = _basis()

    def scalar(data):
        return compress_reduced_basis(
            basis,
            observed=data,
            noise=rhino_bank.noise(),
            epoch_id="e0",
            frozen_tolerance=1e9,
        )(dict(basis.reference_values))

    with pytest.raises(StateValidationError, match="under a trace over the data"):
        jax.grad(scalar)(rhino_bank.observed(jax.random.key(68)))


def test_the_frozen_and_live_densities_agree_at_the_freeze_point_exactly():
    """Zero at the anchor by construction; the residual is what it costs away."""
    basis, data = _basis(), rhino_bank.observed(jax.random.key(65))
    noise = rhino_bank.noise()
    predict, values = rhino_bank.forward()
    reference = basis.rows.T @ basis.c_ref
    frozen_sigma = noise.std(reference)
    live_sigma = noise.std(predict(values))
    # The reference IS the truth for this fixture, so the two sigmas coincide
    # there and the residual comes entirely from probes away from it.
    np.testing.assert_allclose(
        np.asarray(frozen_sigma), np.asarray(live_sigma), rtol=1e-6
    )
    term = compress_reduced_basis(
        basis, observed=data, noise=noise, epoch_id="e0", noise_frozen_at="reference"
    )
    anchor = abs(
        float(
            RawLikelihood(
                predict=basis.predict,
                observed=data,
                sigma=jnp.broadcast_to(frozen_sigma, data.shape),
                names=tuple(values),
                epoch_id="e0",
            )(dict(values))
        )
        - float(
            RawLikelihood(
                predict=basis.predict,
                observed=data,
                sigma=jnp.broadcast_to(live_sigma, data.shape),
                names=tuple(values),
                epoch_id="e0",
            )(dict(values))
        )
    )
    assert anchor < 1e-9
    assert term.frozen_noise_residual > 1e6
