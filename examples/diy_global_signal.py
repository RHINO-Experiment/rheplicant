"""Bring your own 21 cm global signal — the operator, and the two registrations.

The shipped ``GlobalSignalOperator`` says in its own first line that it is a
PLACEHOLDER: a Gaussian trough with a differentiable depth, centre and width,
there so that signal-recovery experiments work end to end before the physics
lands. It is not the package's claim about what the global signal is.

This script replaces it with the EDGES flattened Gaussian (Bowman et al. 2018,
eq. 1) and shows the three things that follow:

1. ``graph_node`` is all the Python API needs. ``assemble`` reads it and drops
   the operator into the same slot the shipped one would have filled.
2. A document reaches it through ``python:``, not through ``type:``.
   ``type:``'s choices come from ``operator_table()``, which walks
   ``rheplicant.radio.__all__`` — the package's own surface — so a class of
   yours is not in it. ``python:`` imports the class by name instead.
3. That route asks for two registrations, and refuses the document until it
   has them. Both refusals are quoted below where they are answered. Neither
   is bureaucracy: a unit the layer cannot convert, or an output dimension it
   cannot check, is how a finite, correctly-shaped, WRONG answer gets past
   everything downstream.

Its parameters are ordinary jax leaves, so the DIY model is a recovery target
exactly like a shipped one — the gradient at the end is the point of the
whole exercise.

Run:  .venv/bin/python examples/diy_global_signal.py
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from rheplicant import Coordinates, Environment, State
from rheplicant.config import run_document
from rheplicant.config.dimensions import (
    DimensionSpec,
    FormulaOperand,
    register_dimension,
    register_dimension_formula,
    signature,
)
from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.radio import ForegroundOperator, GainOperator, assemble

N_TIME, N_FREQ = 8, 32


# ----------------------------------------------------------- the operator --
class FlattenedGaussianSignal(AbstractOperator):
    """T_21 as a flattened Gaussian absorption trough.

    ``-A (1 - exp(-tau e^B)) / (1 - exp(-tau))`` with
    ``B = 4 (nu - nu0)^2 / w^2 * log(-log((1 + e^-tau) / 2) / tau)``.

    ``tau -> 0`` recovers a Gaussian; larger ``tau`` flattens the bottom and
    steepens the walls, which is the shape the EDGES detection reported and
    the shape a plain Gaussian cannot make.

    Attributes:
        amplitude: trough depth [K], positive for absorption.
        centre: trough centre frequency [Hz].
        width: trough full width [Hz].
        flattening: dimensionless flattening ``tau``, > 0.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    #: The only line that ties this class to the signal path. `assemble` and
    #: the config layer both read it; nothing else has to be told.
    graph_node: ClassVar[str] = "global_signal"

    amplitude: jax.Array
    centre: jax.Array
    width: jax.Array
    flattening: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None:
            raise StateValidationError("FlattenedGaussianSignal requires coords.freq.")
        freq = state.coords.freq
        tau = self.flattening
        b = (4.0 * (freq - self.centre) ** 2 / self.width**2) * jnp.log(
            -jnp.log((1.0 + jnp.exp(-tau)) / 2.0) / tau
        )
        profile = -self.amplitude * (1.0 - jnp.exp(-tau * jnp.exp(b))) / (1.0 - jnp.exp(-tau))
        n_time = state.coords.time.shape[0]
        return state.with_data(jnp.broadcast_to(profile[None, :], (n_time, freq.shape[0])))


# ------------------------------------------------- registration one of two --
# Without this, the document is refused with
#
#   dimensions: no dimension selector matches model_field destination
#   'model.global_signal.amplitude'
#
# Every model field says what it is measured in. The alphabet is small on
# purpose and a unit it cannot convert is refused rather than passed through.
_CLASS = f"{FlattenedGaussianSignal.__module__}.{FlattenedGaussianSignal.__qualname__}"
for _field, _unit in (
    ("amplitude", "K"),
    ("centre", "Hz"),
    ("width", "Hz"),
    ("flattening", "dimensionless"),
):
    register_dimension(f"{_CLASS}.{_field}", domain="model_field", dimension=_unit)


# ------------------------------------------------- registration two of two --
# Without this, the document is refused with
#
#   model.global_signal: ... has no registered dimension formula; the plugin
#   must call register_dimension_formula with this concrete class in
#   producers (check A9)
#
# The fields say what goes in; this says what comes OUT, and which fields the
# answer depends on. An operator that quietly emits volts where the graph
# expects kelvin is what check A9 exists to catch.
def _fixed(token: str) -> DimensionSpec:
    return DimensionSpec("fixed", signature(token), unit_policy="inherited")


register_dimension_formula(
    "flattened_gaussian",
    rule="fixed",
    result=_fixed("K"),
    operands=(
        FormulaOperand("amplitude", _fixed("K")),
        FormulaOperand("centre", _fixed("Hz")),
        FormulaOperand("width", _fixed("Hz")),
        FormulaOperand("flattening", _fixed("dimensionless")),
    ),
    producers=(_CLASS,),
)


# --------------------------------------------- 1. through the Python API ---
state = State(
    coords=Coordinates(
        time=jnp.linspace(0.0, 180.0, N_TIME),
        freq=jnp.linspace(60e6, 90e6, N_FREQ),
    ),
    env=Environment(temperature=jnp.array(280.0)),
    key=jax.random.key(0),
    meta={"telescope": "RHINO", "obs_id": "diy-001"},
)
signal = FlattenedGaussianSignal(
    amplitude=jnp.array(0.52),
    centre=jnp.array(78.3e6),
    width=jnp.array(20.7e6),
    flattening=jnp.array(7.0),
)
GAIN = 1.1


def _twin(source: FlattenedGaussianSignal):
    """The same sky and receiver either side of one swapped source."""
    return assemble(
        source,
        ForegroundOperator(
            amplitude=jnp.array(1.0e3), spectral_index=jnp.array(2.5), ref_freq=70e6
        ),
        GainOperator(gain=jnp.array(GAIN)),
    )


trough = signal(state).data[0]
deepest = state.coords.freq[int(trough.argmin())] / 1e6
# Run the whole path twice, once with the trough and once with it flattened to
# nothing. If `assemble` really placed the operator, the difference is the
# trough carried through the gain -- and if it silently dropped it, the
# difference is zero and this line says so.
silent = FlattenedGaussianSignal(
    jnp.array(0.0), signal.centre, signal.width, signal.flattening
)
with_signal = _twin(signal)(state).data
without = _twin(silent)(state).data
contribution = (without - with_signal).max()

print("1. assemble() put it on the signal path from graph_node alone:")
print(f"   trough {float(trough.min()):+.4f} K at {float(deepest):.1f} MHz, "
      f"walls {float(trough[0]):+.4f} / {float(trough[-1]):+.4f} K")
print(f"   swapping it out of the assembled twin moves the output by "
      f"{float(contribution):.4f} K")
print(f"   = the trough through the gain, {abs(float(trough.min())):.2f} "
      f"x {GAIN} = {abs(float(trough.min())) * GAIN:.4f} K")

# --------------------------------------------------- 2. through a document --
# `__main__:` because the class lives in the script you are running. In a real
# project it is `yourpackage.module:YourOperator`, and `plugins: [yourpackage.
# module]` makes the two registrations above run before anything is resolved.
document = {
    "schema_version": 1,
    "runtime": {"seed": 1},
    "observation": {
        "meta": {"telescope": "RHINO"},
        "freq": {
            "grid": {
                "linspace": {"start": 60.0, "stop": 90.0, "num": N_FREQ, "endpoint": True},
                "unit": "MHz",
            }
        },
        "time": {
            "grid": {"arange": {"start": 0.0, "step": 60.0, "num": N_TIME}, "unit": "s"}
        },
    },
    "model": {
        "global_signal": {
            "python": f"__main__:{FlattenedGaussianSignal.__qualname__}",
            "amplitude": {"value": 0.52, "unit": "K"},
            "centre": {"value": 78.3, "unit": "MHz"},
            "width": {"value": 20.7, "unit": "MHz"},
            "flattening": 7.0,
        }
    },
    "runs": [{"name": "forward", "kind": "forward"}],
}
results = run_document(document)
forward = results["forward"]
data = forward.product.data
print()
print("2. the same operator reached from a document through python::")
print(f"   run {forward.name!r} ({forward.kind}) produced {tuple(data.shape)} K, "
      f"minimum {float(data.min()):+.4f} K")
print("   type: could not have named it -- operator_table() walks")
print("   rheplicant.radio.__all__, and this class is not in it.")

# ------------------------------------------------------- 3. it is a target --
# The whole reason to write an operator rather than a lookup table: every
# field is a jax leaf, so a sampler or an optimiser can move it.
def _depth(amplitude: jax.Array) -> jax.Array:
    moved = FlattenedGaussianSignal(amplitude, signal.centre, signal.width, signal.flattening)
    return moved(state).data.min()


gradient = jax.grad(_depth)(jnp.array(0.52))
print()
print("3. its parameters are differentiable leaves, so it is inferrable:")
print(f"   d(trough depth)/d(amplitude) = {float(gradient):+.5f}")
print("   (-1 to within the flattening: the trough bottom IS -amplitude)")
