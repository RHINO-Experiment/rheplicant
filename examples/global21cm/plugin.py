"""The registrations the documents' ``plugins: [global21cm.plugin]`` runs.

A ``python:`` model node is refused until two things are registered for its
class (see ``examples/diy_global_signal.py``): a dimension for every field
the document sets, and a dimension formula for what the operator emits.
Importing this module registers both for the four operators of this
example; nothing else happens at import. The documents use
``CompressedSignal`` alone; the two foreground operators are the forward
models the compression integrates out, registered so they can be
configured and are tested against the simulator.

The module is found through ``PYTHONPATH=examples`` (the directory holding
the ``global21cm`` package); the README's commands set it.
"""

from __future__ import annotations

from rheplicant.config.dimensions import (
    DimensionSpec,
    FormulaOperand,
    register_dimension,
    register_dimension_formula,
    signature,
)

from global21cm.foreground_beamconv import MomentForeground
from global21cm.physical_operator import PhysicalSkyForeground
from global21cm.signal21 import CompressedSignal, Emulated21cmSignal

#: operator class -> (formula name, {field: unit})
FIELDS = {
    MomentForeground: ("moment_foreground", {"coefficients": "K", "basis": "dimensionless"}),
    PhysicalSkyForeground: (
        "physical_sky_foreground",
        {
            "coords": "dimensionless",
            "offset": "K",
            "design": "K",
            "spectra": "dimensionless",
            "response": "dimensionless",
        },
    ),
    Emulated21cmSignal: ("emulated_21cm_signal", {"theta": "dimensionless"}),
    CompressedSignal: ("compressed_signal", {"theta": "dimensionless", "design": "dimensionless"}),
}


def _fixed(token: str) -> DimensionSpec:
    return DimensionSpec("fixed", signature(token), unit_policy="inherited")


def _register(cls, formula: str, fields: dict[str, str]) -> None:
    qualified = f"{cls.__module__}.{cls.__qualname__}"
    for field, unit in fields.items():
        register_dimension(f"{qualified}.{field}", domain="model_field", dimension=unit)
    register_dimension_formula(
        formula,
        rule="fixed",
        result=_fixed("K"),
        operands=tuple(FormulaOperand(name, _fixed(unit)) for name, unit in fields.items()),
        producers=(qualified,),
    )


for _cls, (_formula, _fields) in FIELDS.items():
    _register(_cls, _formula, _fields)
