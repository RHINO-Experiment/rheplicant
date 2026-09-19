"""The foreign-block band, at the offset a real night carries.

``ChainMemory`` refuses a stored block that is not its epoch's own by comparing
the three coefficients of the quadratic form, ``(A, b, c)`` from
``chain._quadratic_form``. The band was ONE number for all three,
``sqrt(eps) * max(|A|, |b|, |c|)``, so the largest coefficient set it for the
other two. Under ``RadiometerNoise`` the constant ``c`` carries the
time-bandwidth product, about 7.2e11 for one RHINO night
(``tests/evidence/conftest.py``), which makes the band about 1.1e4 in every
coefficient. Measured, with two nights sharing one design and differing only in
their data (``b`` 6.8 % apart, ``c`` 74 nats apart): the stack reversed under
the archive was refused at offset 0 and ACCEPTED at offset + 7.2e11 -- the
class's promise that it refuses to be shuffled, broken in the regime it is used
in.

Probed over Gram scales 1e-4..1e8 and constants 0..1e12, the shared band let a
Gram 10 % (and 0.1 %) off through wherever ``|c| / max|A|`` exceeded about 1e7,
and a cross term 19 % off likewise. The band is now per coefficient, each
against the scale of what it is made of and never looser than the shared one:
``max|A|`` for the Gram, ``sqrt(max|A| z.z)`` for the cross term (Cauchy-Schwarz
on ``b = R^T z``), and the shared scale for the constant.

**What stays open.** The constant keeps the shared band, so at ``|c|`` of
7.2e11 two blocks whose constants differ by less than about 1e4 nats are still
told apart only by ``A`` and ``b``. A per-coefficient band cannot tighten that:
the constant's own roundoff scales with it.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.chain import ChainMemory, ornstein_uhlenbeck
from rheplicant.inference.compress import compress_linear
from rheplicant.inference.compressed import QuadraticLikelihood
from rheplicant.inference.sqrtinfo import SqrtInfo
from tests.evidence import chain_bank as bank

ORDER = (*bank.THETA_NAMES, bank.ZETA_NAME)
SHAPES = tuple(() for _ in ORDER)
#: One RHINO night's time-bandwidth product, the offset scalar the evidence
#: layer carries under RadiometerNoise.
RHINO_OFFSET = 7.2e11
SHIFTS = [pytest.param(0.0, id="offset+0"), pytest.param(1e6, id="offset+1e6"),
          pytest.param(RHINO_OFFSET, id="offset+7.2e11")]


def _memory():
    return ChainMemory(bank.factorization(ornstein_uhlenbeck(tau=5.0, sigma=1.0)))


def _night(seed, shift):
    """One night of the SAME design with its own data, offset shifted by ``shift``."""
    A, C, _ = bank.design(0)
    data = np.asarray(jax.random.normal(jax.random.key(100 + seed), (bank.N_SAMPLES,)))
    term = compress_linear(
        design={"t_rx": A[0][:, :1], "gain_slope": A[0][:, 1:], bank.ZETA_NAME: C[0]},
        observed=jnp.asarray(data),
        noise_std=bank.SIGMA,
        shapes={name: () for name in ORDER},
        epoch_id=f"n{seed}",
    )
    return dataclasses.replace(
        term, info=dataclasses.replace(term.info, offset=term.info.offset + shift)
    )


def _term(factor, target, offset, epoch_id="e"):
    info = SqrtInfo(
        factor=jnp.asarray(factor), target=jnp.asarray(target),
        offset=jnp.asarray(float(offset)), names=ORDER, shapes=SHAPES,
    )
    return QuadraticLikelihood(
        info=info, epoch_id=epoch_id, n_observed=bank.N_SAMPLES,
        residual_chi2=jnp.asarray(0.0), residual_dof=0,
    )


def test_the_fixture_is_float64():
    assert _night(0, 0.0).info.factor.dtype == jnp.float64


@pytest.mark.parametrize("shift", SHIFTS)
def test_two_nights_of_one_design_are_honest_in_order(shift):
    """The negative control: the campaign built with remember() is accepted."""
    memory = _memory().remember(_night(0, shift)).remember(_night(1, shift))
    rebuilt = ChainMemory(memory.factorization, memory.stacked, memory._epochs)
    assert tuple(rebuilt.epoch_ids) == ("n0", "n1")


@pytest.mark.parametrize("shift", SHIFTS)
def test_the_same_two_nights_swapped_under_the_archive_are_refused(shift):
    """The regression: accepted at offset + 7.2e11 under the shared band."""
    memory = _memory().remember(_night(0, shift)).remember(_night(1, shift))
    factors, targets, offsets = memory.stacked
    swapped = (factors[::-1], targets[::-1], offsets[::-1])
    with pytest.raises(StateValidationError, match="quadratic forms differ"):
        ChainMemory(memory.factorization, swapped, memory._epochs)


R0 = np.array([[2.0, 0.7, -0.3], [0.0, 1.5, 0.4], [0.0, 0.0, 1.1]])
Z0 = np.array([0.9, -1.3, 0.6])
GRAM_SCALES = [1e-4, 1.0, 1e4, 1e8]
CONSTANTS = [0.0, 1.0, 1e4, 1e8, 1e12]


def _one_block_memory(block_term, epoch_term):
    """A one-epoch chain whose stored block is ``block_term``'s, archived as
    ``epoch_term``: the construction the guard runs on."""
    stored = _memory().remember(block_term)
    return ChainMemory(stored.factorization, stored.stacked, _memory().remember(epoch_term)._epochs)


@pytest.mark.parametrize("gram", GRAM_SCALES)
@pytest.mark.parametrize("constant", CONSTANTS)
class TestEveryScaleRatio:
    """Both outcomes at every cell of the scale grid, called directly."""

    def test_an_honest_block_is_accepted(self, gram, constant):
        epoch = _term(R0 * np.sqrt(gram), Z0 * np.sqrt(gram), constant)
        assert tuple(_one_block_memory(epoch, epoch).epoch_ids) == ("e",)

    def test_a_gram_ten_percent_off_is_refused(self, gram, constant):
        root = np.sqrt(1.1)
        epoch = _term(R0 * np.sqrt(gram), Z0 * np.sqrt(gram), constant)
        foreign = _term(R0 * np.sqrt(gram) * root, Z0 * np.sqrt(gram) / root, constant)
        with pytest.raises(StateValidationError, match="quadratic forms differ"):
            _one_block_memory(foreign, epoch)

    def test_a_cross_term_ten_percent_off_is_refused(self, gram, constant):
        """``z`` moved by 10 % of its norm, the offset moved with ``z.z`` so the
        constant is unchanged: only ``b`` differs."""
        target = Z0 * np.sqrt(gram)
        moved = target + 0.1 * np.linalg.norm(target) * np.array([1.0, 0.0, 0.0])
        epoch = _term(R0 * np.sqrt(gram), target, constant)
        foreign = _term(
            R0 * np.sqrt(gram), moved, constant + 0.5 * (moved @ moved - target @ target)
        )
        with pytest.raises(StateValidationError, match="quadratic forms differ"):
            _one_block_memory(foreign, epoch)
