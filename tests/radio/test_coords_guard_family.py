"""One guard, copy-pasted nine times, tested three times.

Nine operators open ``__call__`` with the same sentence -- *"X requires
state.coords with time and freq axes"* -- and a full-suite coverage run found
six of the nine ``raise`` lines never executed. The three that were covered are
the three whose files someone happened to be working in.

That is the shape this repository keeps rediscovering: a guard with N copies
gets a test written for the copies that were in front of somebody. The fix is
not six more hand-written tests, which would leave copy ten in the same
position. It is to derive the population from the source, so that adding the
guard to a new operator without covering it fails here.

What this file asserts, in order:

1. every operator carrying the sentence is in ``COORDS_GUARDED`` (derived from
   the source, so a tenth copy cannot be added silently);
2. each one actually raises on a coords-less state;
3. each names ITSELF, so nine identical refusals stay distinguishable -- the
   sentence is the operator's name plus a fixed suffix, and a copy-paste that
   forgot to change the name would otherwise pass every check above.
"""

import inspect
import re

import jax.numpy as jnp
import pytest

import rheplicant.radio as radio
from rheplicant.core.errors import StateValidationError
from rheplicant.core.state import State
from rheplicant.radio.sky import PowerLawSkyModel

#: The sentence that defines the family. Matched against each operator's own
#: source rather than against a list, so the list below is checked and not
#: trusted.
_GUARD_SENTENCE = re.compile(r"requires state\.coords with time and freq")

#: Minimal constructor arguments, one per family member. Values are arbitrary
#: but distinct -- nothing here should depend on them, and if a future test
#: does, distinct values make the dependence visible instead of accidental.
COORDS_GUARDED: dict[str, dict] = {
    "AtmosphericEmissionOperator": {"t_atm": jnp.array(3.0)},
    "BasisTemperatureOperator": {
        "coeff": jnp.ones((2, 2)),
        "time_basis": jnp.ones((4, 2)),
        "freq_basis": jnp.ones((8, 2)),
    },
    "CalLoadOperator": {"t_load": jnp.array(300.0)},
    "ForegroundOperator": {
        "amplitude": jnp.array(120.0),
        "spectral_index": jnp.array(-2.6),
        "ref_freq": 70e6,
    },
    "GlobalSignalOperator": {
        "depth": jnp.array(-0.2),
        "centre": jnp.array(78e6),
        "width": jnp.array(19e6),
    },
    "GroundPickupOperator": {"coupling": jnp.array(0.05), "t_ground": jnp.array(290.0)},
    "PointSourceOperator": {"level": jnp.array(7.0)},
    "RFIOperator": {"amplitude": jnp.array(11.0), "occupancy": 0.1},
    "SkySourceOperator": None,  # built below; it needs a model and a projector
}


def _sky_source():
    from rheplicant.radio.sky import MatrixProjector

    return radio.SkySourceOperator(
        # Per-pixel amplitudes all different, and a projector that is not a
        # uniform average: a symmetric sky through a flat projector would give
        # the same answer under a transposed or partially applied projection.
        sky_model=PowerLawSkyModel(
            amplitude=jnp.arange(1.0, 13.0),
            spectral_index=jnp.array(-2.6),  # scalar: the model is per-pixel
            ref_freq=70e6,                   # in amplitude only
            n_pix=12,
        ),
        projector=MatrixProjector(
            matrix=jnp.linspace(0.5, 1.5, 48).reshape(4, 12) / 12.0
        ),
    )


def _operator(name: str):
    if COORDS_GUARDED[name] is None:
        return _sky_source()
    return getattr(radio, name)(**COORDS_GUARDED[name])


def _carrying_the_guard() -> set[str]:
    """Every exported operator whose own source contains the sentence."""
    found = set()
    for name in radio.__all__:
        obj = getattr(radio, name)
        if not inspect.isclass(obj):
            continue
        try:
            source = inspect.getsource(obj)
        except (OSError, TypeError):  # pragma: no cover - not reachable here
            continue
        if _GUARD_SENTENCE.search(source):
            found.add(name)
    return found


def test_the_table_is_the_family_and_the_family_is_the_table():
    """The assertion that makes the rest of this file self-maintaining.

    Derived from the source, so the tenth copy of the guard fails here with a
    message naming itself, rather than joining the six that had no test.
    """
    assert _carrying_the_guard() == set(COORDS_GUARDED), {
        "carry the guard but are untested": sorted(
            _carrying_the_guard() - set(COORDS_GUARDED)
        ),
        "listed but no longer carry it": sorted(
            set(COORDS_GUARDED) - _carrying_the_guard()
        ),
    }


@pytest.mark.parametrize("name", sorted(COORDS_GUARDED))
def test_a_coords_less_state_is_refused(name):
    with pytest.raises(StateValidationError):
        _operator(name)(State(data=jnp.zeros((4, 8))))


@pytest.mark.parametrize("name", sorted(COORDS_GUARDED))
def test_the_refusal_names_the_operator_that_raised_it(name):
    """Nine identical sentences have to stay distinguishable.

    The message is the class name plus a fixed suffix, so a copy-paste that
    brought the guard across without changing the name would raise, would be
    a ``StateValidationError``, and would satisfy the test above -- while
    telling a user to go and look at the wrong operator.
    """
    with pytest.raises(StateValidationError) as excinfo:
        _operator(name)(State(data=jnp.zeros((4, 8))))
    assert name in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("name", sorted(COORDS_GUARDED))
def test_a_state_with_coords_gets_past_this_guard(name):
    """The other branch.

    Without it, an operator whose ``__call__`` began with an unconditional
    raise would pass every test above. That is not hypothetical here: the
    guard's condition is a three-way ``or``, so a mutation collapsing it to
    ``True`` is one character.
    """
    import jax

    from rheplicant.core.coordinates import Coordinates

    # A key, because RFIOperator draws: it is one of the package's two
    # stochastic operators, and its coords guard sits ahead of the draw. Giving
    # every member a key keeps the parametrization uniform rather than special
    # -casing the one that needs it, which is how the six untested copies came
    # to be untested in the first place.
    state = State(
        data=jnp.zeros((4, 8)),
        coords=Coordinates(
            time=jnp.arange(4.0), freq=jnp.linspace(60e6, 80e6, 8)
        ),
        key=jax.random.key(0),
    )
    out = _operator(name)(state)
    assert out.data is not None
    assert out.data.shape == (4, 8)
