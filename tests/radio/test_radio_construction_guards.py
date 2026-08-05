"""Three radio constructors that refuse a setting nothing could mean.

The radio half of the construction-guard batch. The directory prefix in the
basename is load-bearing: ``tests/`` deliberately has no ``__init__.py``, so
pytest imports test modules by bare basename and two files named
``test_construction_guards.py`` in two directories cannot both be collected --
an ``EXIT=2`` at collection that appears only when both are in one run, and so
survives a per-file check.

Its companion, ``tests/inference/test_inference_construction_guards.py``,
carries the argument for why
this family is tested case by case rather than from a table derived over all 31
classes with a raising ``__check_init__`` -- and carries the one derived check
that IS worth having here: ``ref_freq must be > 0`` and ``n_pix must be a
positive int`` are each written more than once, and the census that keeps their
copies enumerated lives there because it is a source scan over the whole
package, not over one directory.

Two of the three guards are the ones a copy census cannot help with anyway.
``cg_maxiter`` exists once. And the ``n_pix`` copy covered here is
``PowerLawSkyModel``'s: ``UniformSkyModel``'s identical copy was already
covered, which is exactly the split the census exists to stop widening.
"""

import math

import jax.numpy as jnp
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.core.state import State
from rheplicant.radio.environment.ionosphere import IonosphereOperator
from rheplicant.radio.filters.skyspace import SkySpaceFilter
from rheplicant.radio.sky.model import PowerLawSkyModel
from rheplicant.radio.sky.projection import MatrixProjector

#: A projector that is neither square nor uniform, so a transposed or
#: partially-applied projection would give a different answer.
PROJECTOR = MatrixProjector(matrix=jnp.linspace(0.25, 1.75, 12).reshape(4, 3))


# --------------------------------------------------------------------------
# IonosphereOperator.ref_freq -- the untested third copy of a tripled guard.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ref_freq", [0.0, -70e6])
def test_a_non_positive_reference_frequency_is_refused(ref_freq):
    """``0.0`` is the excluded endpoint, and the one a caller reaches by default.

    ``ref_freq`` divides the frequency axis, so zero is not merely meaningless:
    it produces an infinity that the ``**-2`` then turns back into a finite
    zero-distortion factor. The refusal is what keeps that from looking like a
    working model with the ionosphere switched off.
    """
    with pytest.raises(StateValidationError) as excinfo:
        IonosphereOperator(delta=jnp.array(0.05), ref_freq=ref_freq)
    assert "ref_freq must be > 0" in str(excinfo.value)


def test_the_construction_guard_is_not_the_coords_guard(coords):
    """``IonosphereOperator`` raises ``StateValidationError`` in two places.

    One refuses the constructor argument, the other refuses a state with no
    frequency axis. Both are the same exception type, so a test matching only
    the type could be satisfied by the wrong one.
    """
    with pytest.raises(StateValidationError) as construction:
        IonosphereOperator(delta=jnp.array(0.05), ref_freq=0.0)
    with pytest.raises(StateValidationError) as call_time:
        IonosphereOperator(delta=jnp.array(0.05), ref_freq=70e6)(
            State(data=jnp.zeros((8, 4)))
        )

    assert "ref_freq must be > 0" in str(construction.value)
    assert "ref_freq must be > 0" not in str(call_time.value)
    assert "requires state.coords.freq" in str(call_time.value)
    assert "requires state.coords.freq" not in str(construction.value)
    assert coords.freq is not None  # the axis the second refusal is about


def test_a_positive_reference_frequency_is_accepted_and_used(coords):
    """The other branch, and it has to be the branch that does the arithmetic.

    ``delta`` is chosen large enough that the chromatic factor is far from 1,
    and the check is made at a frequency away from ``ref_freq`` so a guard that
    silently substituted a default ``ref_freq`` would not match.
    """
    operator = IonosphereOperator(delta=jnp.array(0.5), ref_freq=1e-9)
    assert operator.ref_freq == 1e-9  # any positive value, however small

    operator = IonosphereOperator(delta=jnp.array(0.5), ref_freq=70e6)
    state = State(data=jnp.ones((coords.time.shape[0], coords.freq.shape[0])), coords=coords)
    out = operator(state)
    expected = 1.0 + 0.5 * (coords.freq / 70e6) ** (-2.0)
    assert jnp.allclose(out.data, expected[None, :])
    assert not jnp.allclose(out.data, 1.0)


# --------------------------------------------------------------------------
# SkySpaceFilter.cg_maxiter -- a single guard, no copies.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cg_maxiter", [0, -5, 100.0, "100"])
def test_a_cg_iteration_cap_that_is_not_a_positive_int_is_refused(cg_maxiter):
    """Both branches of one condition: the type check and the size check.

    ``100.0`` is the interesting one -- it is a perfectly sensible number that
    would reach ``jax.scipy.sparse.linalg.cg`` as a float ``maxiter``.
    """
    with pytest.raises(StateValidationError) as excinfo:
        SkySpaceFilter(
            projector=PROJECTOR, regularization=jnp.array(1e-3), cg_maxiter=cg_maxiter
        )
    assert "cg_maxiter must be a positive int" in str(excinfo.value)


def test_the_cg_guard_and_the_projector_guard_say_different_things():
    """Two ``StateValidationError``s from the same ``__check_init__``."""
    with pytest.raises(StateValidationError) as bad_maxiter:
        SkySpaceFilter(projector=PROJECTOR, regularization=jnp.array(1e-3), cg_maxiter=0)
    with pytest.raises(StateValidationError) as bad_projector:
        SkySpaceFilter(projector=object(), regularization=jnp.array(1e-3))

    assert "cg_maxiter must be a positive int" in str(bad_maxiter.value)
    assert "cg_maxiter must be a positive int" not in str(bad_projector.value)
    assert "must be an AbstractSkyProjector" in str(bad_projector.value)
    assert "must be an AbstractSkyProjector" not in str(bad_maxiter.value)


def test_one_iteration_is_accepted():
    """The boundary: the smallest cap the guard admits."""
    filt = SkySpaceFilter(projector=PROJECTOR, regularization=jnp.array(1e-3), cg_maxiter=1)
    assert filt.cg_maxiter == 1


# --------------------------------------------------------------------------
# PowerLawSkyModel.n_pix -- the untested copy of a doubled guard.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_pix", [0, -3, 12.0, None])
def test_a_pixel_count_that_is_not_a_positive_int_is_refused(n_pix):
    with pytest.raises(StateValidationError) as excinfo:
        PowerLawSkyModel(
            amplitude=jnp.arange(1.0, 4.0),
            spectral_index=jnp.array(-2.6),
            ref_freq=70e6,
            n_pix=n_pix,
        )
    assert "n_pix must be a positive int" in str(excinfo.value)


def test_the_pixel_guard_and_the_frequency_guard_say_different_things():
    """``PowerLawSkyModel`` carries both copied guards, and they are one type."""
    with pytest.raises(StateValidationError) as bad_pixels:
        PowerLawSkyModel(
            amplitude=jnp.arange(1.0, 4.0),
            spectral_index=jnp.array(-2.6),
            ref_freq=70e6,
            n_pix=0,
        )
    with pytest.raises(StateValidationError) as bad_ref_freq:
        PowerLawSkyModel(
            amplitude=jnp.arange(1.0, 4.0),
            spectral_index=jnp.array(-2.6),
            ref_freq=0.0,
            n_pix=3,
        )

    assert "n_pix must be a positive int" in str(bad_pixels.value)
    assert "n_pix must be a positive int" not in str(bad_ref_freq.value)
    assert "ref_freq must be > 0" in str(bad_ref_freq.value)
    assert "ref_freq must be > 0" not in str(bad_pixels.value)


def test_a_single_pixel_sky_is_accepted_and_evaluates(coords):
    """The boundary the guard admits, evaluated rather than only constructed.

    One pixel is the smallest sky the model can describe, and the spectrum it
    produces still has to be the power law: a guard whose condition was inverted
    would refuse this, and nothing in the refusal tests above would notice.
    """
    model = PowerLawSkyModel(
        amplitude=jnp.array([9.0]),
        spectral_index=jnp.array(-2.6),
        ref_freq=70e6,
        n_pix=1,
    )
    emission = model(coords.freq)
    assert emission.shape == (coords.freq.shape[0], 1)
    expected = 9.0 * (coords.freq / 70e6) ** 2.6
    assert jnp.allclose(emission[:, 0], expected)


def test_several_pixels_stay_distinguishable(coords):
    """An asymmetric amplitude map, so a collapsed or averaged sky would show."""
    model = PowerLawSkyModel(
        amplitude=jnp.array([2.0, 5.0, 11.0]),
        spectral_index=jnp.array(-2.6),
        ref_freq=70e6,
        n_pix=3,
    )
    emission = model(coords.freq)
    assert emission.shape == (coords.freq.shape[0], 3)
    assert len(set(emission[0].tolist())) == 3


# --------------------------------------------------------------------------
# The finding, radio half.
# --------------------------------------------------------------------------


def test_a_nan_reference_frequency_is_accepted_and_poisons_the_band(coords):
    """``nan <= 0`` is False, so the guard does not fire.

    Pins a gap, not a contract -- see the class of the same purpose in
    ``tests/inference/test_inference_construction_guards.py`` for the fix
    (``if not self.ref_freq > 0``) and for the two places in this package that
    already use the NaN-safe form. This guard is copy-pasted three times, so the
    gap is in ``ForegroundOperator`` and ``PowerLawSkyModel`` too; when the
    inversion lands, move ``nan`` into the refusal parametrizations above and
    delete this.
    """
    operator = IonosphereOperator(delta=jnp.array(0.5), ref_freq=float("nan"))
    assert math.isnan(operator.ref_freq)
    state = State(data=jnp.ones((coords.time.shape[0], coords.freq.shape[0])), coords=coords)
    assert jnp.all(jnp.isnan(operator(state).data))


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (
            lambda: SkySpaceFilter(
                projector=PROJECTOR, regularization=jnp.array(1e-3), cg_maxiter=float("nan")
            ),
            "cg_maxiter must be a positive int",
        ),
        (
            lambda: PowerLawSkyModel(
                amplitude=jnp.arange(1.0, 4.0),
                spectral_index=jnp.array(-2.6),
                ref_freq=70e6,
                n_pix=float("nan"),
            ),
            "n_pix must be a positive int",
        ),
    ],
    ids=["cg_maxiter", "n_pix"],
)
def test_the_int_typed_settings_refuse_nan(build, expected):
    """These two do refuse NaN -- because ``nan`` is a float, not because of the
    comparison. The type check runs first, so the size comparison beneath it is
    never reached and its blind spot never matters here."""
    with pytest.raises(StateValidationError) as excinfo:
        build()
    assert expected in str(excinfo.value)
