"""T1, against T0. Absolute log-densities, never moments.

The stored term is ``O(n_S^2)`` and holds no ``n_data``-sized array: the
dictionary is one object shared by the whole campaign, and this is where that
claim is checked rather than asserted in a docstring.

**Why absolute.** T1's offset carries two constants -- the masked Gaussian
normalisation and the QR corner ``-0.5 rho^2`` -- and neither has any effect on
the shape of the posterior, its gradient or its curvature. Plan A shipped both
errors once, in the linear tier, and both survived probes built on moments.
Measured on this fixture, one epoch, 128 channels:

    normalisation  +200.738 nats
    QR corner       -51.321 nats
    offset          +149.418 nats
    T1 - T0 at five probes one prior sigma out, worst  1.3e-6 nats

So a comparison at a thousandth of a nat is 40 000 times below the smaller
constant and 770 times above the truncation gap. It can see a dropped term and
it cannot be fooled by the approximation this tier is.

``n_basis = 6`` throughout, and not by taste: the whitened bank's numerical rank
on this fixture is 6, so ``build_reduced_basis`` refuses anything larger. That
is enough -- the four score directions are seeded, so the span contains the
tangent space by construction, and the residual the truncation leaves at these
probes is ``||(I - Pi) mu||`` of ``1.6e-6`` to ``6.6e-5`` in whitened units.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compress import compress_reduced_basis
from rheplicant.inference.compressed import COEFFICIENTS, RawLikelihood
from rheplicant.inference.reduced_basis import build_reduced_basis
from tests.evidence import rhino_bank

#: The whitened bank's numerical rank on this fixture, measured and pinned in
#: `test_reduced_basis_build.py`. A larger n_basis is refused, not merely
#: wasteful.
N_BASIS = 6

#: Measured, on the epoch below. The two pure constants in the offset, and the
#: worst absolute gap to T0 over the five probes.
NORMALISATION = 200.738482
QR_CORNER = -51.320714
WORST_GAP_NATS = 1.3e-6


@pytest.fixture(scope="module")
def basis():
    return build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(jax.random.key(20)),
        n_basis=N_BASIS,
        support=rhino_bank.support(),
    )


@pytest.fixture(scope="module")
def epoch():
    return rhino_bank.observed(jax.random.key(21))


def _term(basis, observed, **overrides):
    keywords = {
        "noise": rhino_bank.noise(),
        "epoch_id": "e0",
        "noise_frozen_at": "reference",
        **overrides,
    }
    return compress_reduced_basis(basis, observed=observed, **keywords)


def _oracle(observed, **overrides):
    """T0 on the same epoch, with sigma frozen at the same reference."""
    predict, values = rhino_bank.forward()
    keywords = {
        "sigma": rhino_bank.noise().std(predict(values)),
        "names": tuple(rhino_bank.TRUTH),
        "epoch_id": "e0",
        "noise_frozen_at": "reference",
        **overrides,
    }
    return RawLikelihood(predict=predict, observed=observed, **keywords)


def _probes(count=5):
    """Draws one prior sigma from the truth -- inside `rhino_bank.support()`."""
    _, values = rhino_bank.forward()
    keys = jax.random.split(jax.random.key(22), count)
    return [
        {
            name: values[name]
            + rhino_bank.PRIOR_STD[name]
            * jax.random.normal(jax.random.fold_in(key, i), ())
            for i, name in enumerate(values)
        }
        for key in keys
    ]


def test_t1_reproduces_t0_absolutely_where_the_span_holds_the_model(basis, epoch):
    """Not 'up to a constant': the offset is part of the answer.

    A tenth of a milli-nat. The two constants a compressor forgets are 51 and
    201 nats, so this fails loudly for either and passes for the truncation.
    """
    term, oracle = _term(basis, epoch), _oracle(epoch)
    gaps = [float(term(probe)) - float(oracle(probe)) for probe in _probes()]
    assert max(abs(gap) for gap in gaps) < 1e-3, gaps


def test_the_gap_to_t0_is_the_truncation_and_it_is_this_small(basis, epoch):
    """The pin. Loosening this number is a statement about the basis, not a knob."""
    term, oracle = _term(basis, epoch), _oracle(epoch)
    worst = max(abs(float(term(p)) - float(oracle(p))) for p in _probes())
    assert worst < 10 * WORST_GAP_NATS, worst


def test_on_noiseless_data_at_the_truth_the_density_is_exactly_the_normalisation(
    basis,
):
    """Both constants, isolated, with the chi-square driven to zero.

    Data equal to the reference prediction leaves nothing for the coefficients
    to fit, so the QR corner is zero and T1's whole density is the masked
    Gaussian normalisation. A term missing that term returns 0.0 here and a term
    missing the corner is indistinguishable -- which is exactly why the noisy
    epoch above is tested too, and why neither test alone would do.
    """
    predict, values = rhino_bank.forward()
    clean = predict(values)
    term = _term(basis, clean, epoch_id="clean")
    assert float(term(values)) == pytest.approx(NORMALISATION, abs=1e-5)
    assert float(term(values)) == pytest.approx(float(_oracle(clean)(values)), abs=1e-9)


def test_the_offset_is_the_normalisation_minus_the_qr_corner(basis, epoch):
    """Both constants, named and measured, so a drop is attributable.

    The QR corner is the part of the whitened residual no coefficient can reach.
    It is constant in theta, so dropping it moves every density by the same
    -51.3 nats and moves no gradient at all.
    """
    term = _term(basis, epoch)
    assert float(term.info.offset) == pytest.approx(
        NORMALISATION + QR_CORNER, abs=1e-5
    )


def test_the_stored_term_holds_nothing_of_size_n_data(basis, epoch):
    """The storage claim: O(n_S^2) per epoch, and the dictionary shared once."""
    term = _term(basis, epoch)
    own = [np.asarray(leaf) for leaf in jax.tree.leaves(term.info)]
    assert own, "the term stores no arrays at all, which cannot be right"
    assert max(leaf.size for leaf in own) <= basis.n_basis * basis.n_basis
    assert term.basis is basis  # a reference, not a copy


def test_the_term_is_approximate_and_therefore_carries_a_support(basis, epoch):
    term = _term(basis, epoch)
    assert term.exact is False
    assert set(term.support) == set(rhino_bank.TRUTH)
    assert term.info.names == (COEFFICIENTS,)


def test_the_latents_are_in_declared_order_not_flatten_order(basis, epoch):
    """`jax` sorts a dict's keys; this fixture's latents are not sorted.

    Sorted order here is ("amplitude", "index", "running", "t21_depth"), so a
    term that reported the flattened order would be wrong by a permutation --
    and `BayesMemory.remember` compares this tuple against the factorization's
    declared order before folding the term irreversibly into the QR.
    """
    assert _term(basis, epoch).latents == ("t21_depth", "index", "amplitude", "running")


def test_a_basis_with_no_support_is_refused(epoch):
    naked = build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(jax.random.key(23)),
        n_basis=N_BASIS,
    )
    with pytest.raises(StateValidationError, match="support"):
        _term(naked, epoch)


def test_a_support_passed_to_the_call_is_accepted_which_the_refusal_must_not_eat(
    epoch,
):
    """The nearest legitimate case. A guard that over-refuses is its own bug."""
    naked = build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(jax.random.key(23)),
        n_basis=N_BASIS,
    )
    term = _term(naked, epoch, support=rhino_bank.support())
    assert set(term.support) == set(rhino_bank.TRUTH)


def test_compression_refuses_to_run_under_jit(basis, epoch):
    """n_observed is a Python int on the term, so it cannot be traced.

    Sigma here comes from the basis's reference prediction rather than from the
    data, so this looked as though it might be jittable where `compress_linear`
    is not. It is not, and the reason is omnistaging: inside a jit trace every
    jnp call is staged out whether or not its operands are traced. Grad and vmap
    lift only what is actually traced, which is why they still work below.
    """
    with pytest.raises(StateValidationError, match="cannot run under jit"):
        jax.jit(lambda d: _term(basis, d).info.offset)(epoch)


def test_traced_flags_are_refused_too(basis, epoch):
    """The case the guard was written for: the unflagged count really is unknown."""
    from rheplicant.inference.noise import FlaggedNoise

    def go(flags):
        return _term(
            basis, epoch, noise=FlaggedNoise(rhino_bank.noise(), flags)
        ).info.offset

    with pytest.raises(StateValidationError, match="cannot run under jit"):
        jax.jit(go)(jnp.zeros(rhino_bank.N_FREQ, bool))


def test_grad_over_the_data_still_works(basis, epoch):
    probe = _probes(1)[0]

    def scalar(data):
        return _term(basis, data)(probe)

    gradient = jax.grad(scalar)(epoch)
    assert gradient.shape == epoch.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_vmap_over_a_stack_of_epochs_still_works(basis, epoch):
    probe = _probes(1)[0]
    stack = jnp.stack([epoch, rhino_bank.observed(jax.random.key(24))])
    values = jax.vmap(lambda d: _term(basis, d)(probe))(stack)
    assert values.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(values)))


def test_a_flagged_epoch_gives_a_finite_term_and_a_smaller_n_observed(basis, epoch):
    """And the NaN is at a flagged sample, which is the usual reason it is flagged.

    `0.0 * nan` is `nan`, so a mask that multiplies rather than selects poisons
    `target` and `offset` while leaving `factor` finite -- every density NaN, and
    every conditioning diagnostic clean. Two tasks in a row found that bug in
    this subsystem, so the poisoned sample here is deliberate.
    """
    from rheplicant.inference.noise import FlaggedNoise

    flags = jnp.zeros(rhino_bank.N_FREQ, bool).at[jnp.array([3, 40, 41])].set(True)
    poisoned = epoch.at[40].set(jnp.nan)
    term = _term(
        basis, poisoned, noise=FlaggedNoise(rhino_bank.noise(), flags), epoch_id="e1"
    )
    assert term.n_observed == rhino_bank.N_FREQ - 3
    assert bool(jnp.all(jnp.isfinite(term.info.factor)))
    assert bool(jnp.all(jnp.isfinite(term.info.target)))
    assert np.isfinite(float(term.info.offset))
    assert np.isfinite(float(term(_probes(1)[0])))


def test_a_nan_sigma_is_refused_rather_than_read_as_a_flag(basis, epoch):
    """`inf` means 'not observed'; NaN means the noise model is broken."""

    class _BrokenNoise:
        depends_on_prediction = True

        def std(self, prediction):
            return jnp.asarray(prediction).at[7].set(jnp.nan)

    with pytest.raises(StateValidationError, match="NaN"):
        _term(basis, epoch, noise=_BrokenNoise())


def test_a_memory_cannot_archive_a_t1_term(basis, epoch, tmp_path):
    from rheplicant.inference.archive import save_memory
    from rheplicant.inference.factorize import Factorization
    from rheplicant.inference.memory import BayesMemory

    memory = BayesMemory(
        Factorization(rhino_bank.space()), archive=(_term(basis, epoch),)
    )
    with pytest.raises(StateValidationError, match="reconstruction spec"):
        save_memory(memory, tmp_path / "campaign.eqx")
