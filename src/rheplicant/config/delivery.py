"""Delivery: the destination field decides how a resolved value arrives.

A config value is a number in a document. Whether it reaches an operator as a
Python ``int`` or as a traced ``jnp`` array is not the document's choice and
not this layer's -- it is written on the target class, in the ``dataclasses``
field metadata equinox populates. So delivery reads the class first.

Four measurements are the whole argument for this module.

1. ``ADCOperator(n_bits=jnp.asarray(12))`` warns ``A JAX array is being set as
   static!`` and then raises. ``ForegroundOperator(ref_freq=jnp.asarray(1.4e8))``
   only warns: it *constructs*, the forward numbers are bit-identical, and
   ``eqx.filter_grad`` then returns ``1.4e+08`` where a gradient belongs.
   ``FlaggingOperator.threshold`` has no ``__check_init__`` at all and takes a
   whole array, detonating later at an unrelated pytree comparison.
2. ``AntennaLossOperator(efficiency=1)`` stores ``int32``. An integer array is
   not an *inexact* array, so ``eqx.partition(op, eqx.is_inexact_array)``
   returns ``[]`` and the field is silently untrainable. A YAML ``1`` and a
   YAML ``1.0`` must not differ in what can be inferred.
3. A YAML sequence is a Python ``list``. On a static ``tuple`` field it
   constructs fine and makes the module unhashable.
4. ``astype("float64")`` returns ``float32`` when ``jax_enable_x64`` is off,
   with no warning, and every later dtype check then agrees with the downcast
   value. The flag is process-global and must be set before any array exists,
   so a document cannot make float64 true merely by asking for it -- and
   ``general_pointing.py`` puts the cost of getting this wrong at O(10%).

The model is ``CWCalibrationOperator``'s converters
(``radio/instrument/calibration.py:242-255``): coerce to a clean static scalar
*before* equinox's static check runs, and refuse with a message rather than a
warning. This module does the same thing one step earlier.
"""

import dataclasses
import numbers
import types
import typing
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from _rheplicant_bootstrap.types import DestinationDescriptor, Origin
from rheplicant.config.context import ResolutionContext
from rheplicant.config.dimensions import DimensionSignature, dimension_for, dimension_of
from rheplicant.config.errors import ConfigError
from rheplicant.config.units import Unit, canonical_unit
from rheplicant.core.frozen import FrozenMapping

#: The values ``as:`` may take, and what each claims about the destination.
DELIVERY_MODES: tuple[str, ...] = (
    "traced",
    "static_int",
    "static_float",
    "static_str",
    "static_bool",
    "static_tuple",
    "static_mapping",
)

#: Forms that produce an array. None of them can land on a static field.
ARRAY_FORMS: frozenset[str] = frozenset(
    {
        "zeros",
        "ones",
        "full",
        "list",
        "linspace",
        "arange",
        "modulo",
        "from_grid",
        "basis_fit",
        "normal",
        "uniform",
        "file",
        "stack",
    }
)


class FieldSpec(NamedTuple):
    """What a destination field says about itself.

    A bullet list rather than an ``Attributes:`` section, for the reason given
    on :class:`rheplicant.config.units.Unit`: napoleon's copy of a NamedTuple's
    fields duplicates the one autodoc already emits.

    * ``name`` -- the Python field name, also what a refusal quotes.
    * ``annotation`` -- the resolved type object (``int``, ``float``,
      ``jax.Array``…).
    * ``static`` -- ``True`` when equinox will put this field in the treedef.
    * ``converter`` -- the field's own converter, or ``None``.
    * ``required`` -- no default and no default_factory.
    """

    name: str
    annotation: Any
    static: bool
    converter: Callable[[Any], Any] | None
    required: bool


def field_specs(cls: type) -> dict[str, FieldSpec]:
    """Every ``init`` field of an ``eqx.Module`` subclass, by name.

    ``typing.get_type_hints`` rather than ``f.type``: four modules in the
    package use ``from __future__ import annotations``, and although none of
    them currently defines an ``eqx.Module``, a string annotation would make
    every branch below fall through to "traced" silently.
    """
    hints = typing.get_type_hints(cls)
    return {
        f.name: FieldSpec(
            name=f.name,
            annotation=hints.get(f.name, f.type),
            static=f.metadata.get("static", False) is True,
            converter=f.metadata.get("converter"),
            required=(
                f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
            ),
        )
        for f in dataclasses.fields(cls)
        if f.init
    }


def _optional_payload(annotation: Any) -> tuple[Any, bool]:
    """Reduce ``T | None`` to ``(T, True)``. Anything else is ``(it, False)``.

    ``field_specs`` resolves annotations with ``typing.get_type_hints``, so a
    field written ``int | None`` arrives as a ``types.UnionType`` rather than
    as ``int``. Classified by identity against ``bool``/``int``/``float``/
    ``str`` it matches nothing, and before this reduction existed it fell
    through to ``static_other`` -- the mode reserved for objects the value
    grammar cannot express, which delivers its payload UNTOUCHED. Measured on
    ``DriftScanProjector.freq_chunk`` (``int | None``): ``true`` was delivered
    as Python ``True``, a numpy scalar was delivered as a numpy scalar, and
    the string ``"eight"`` was delivered as a string -- while the same three
    values written to the plainly-typed ``mask_iterations: int`` beside it
    were each refused by name. The field was not weakly guarded; it had no
    guard at all.

    ``None`` stays legal, because that is what the annotation says: the
    reduction narrows how a NON-``None`` value is checked and takes nothing
    away.
    """
    origin = typing.get_origin(annotation)
    if origin is not types.UnionType and origin is not typing.Union:
        return annotation, False
    members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    if len(members) == 1:
        return members[0], True
    return annotation, False


def mode_of(spec: FieldSpec) -> str:
    """The delivery mode a field's own declaration implies."""
    if not spec.static:
        return "traced"
    annotation, _ = _optional_payload(spec.annotation)
    if typing.get_origin(annotation) in (types.UnionType, typing.Union):
        # A genuine multi-member union such as ``int | str``. Not refused here
        # -- mode_of is called to ENUMERATE fields (the schema and the GUI walk
        # every field of every operator), and a package-authoring mistake must
        # not make enumeration explode. It is refused in deliver_checked, at
        # the moment something is actually written to it.
        return "static_union"
    # Identity, not isinstance: bool subclasses int, so issubclass(bool, int)
    # is True and an isinstance-style test would classify a bool field as an
    # int one. Identity checks are mutually exclusive, so the order of the
    # four below carries no meaning -- the ordering that does is in
    # _as_static_int, where the guard is isinstance and bool must come first.
    if annotation is bool:
        return "static_bool"
    if annotation is int:
        return "static_int"
    if annotation is float:
        return "static_float"
    if annotation is str:
        return "static_str"
    origin = typing.get_origin(annotation) or annotation
    if origin is tuple:
        return "static_tuple"
    if isinstance(origin, type) and issubclass(origin, Mapping):
        return "static_mapping"
    return "static_other"


def declared_or_inferred_mode(
    context: ResolutionContext,
    destination: DestinationDescriptor,
    spec: FieldSpec,
    declared: str | None,
) -> str | None:
    """Return the exact mode delivery consumes, recording an omitted one."""
    if declared is not None:
        return declared
    return context.use_default(
        f"{destination.document_path}.as",
        mode_of(spec),
    )


def _refuse_array_form(spec: FieldSpec, source: str) -> None:
    raise ConfigError(
        f"Field {spec.name!r} is static -- equinox puts it in the treedef, where it "
        f"is part of the jit cache key -- and a {source!r} form produces an array. "
        "Measured, this fails in three different ways depending on the field: "
        "ADCOperator(n_bits=Array(12)) warns 'A JAX array is being set as static!' "
        "and then raises; ForegroundOperator(ref_freq=Array(...)) only warns, so it "
        "constructs, the forward numbers are unchanged, and filter_grad hands back "
        "the static value where a gradient belongs; FlaggingOperator.threshold has no "
        "check at all and detonates later at an unrelated pytree comparison. Write a "
        "single number here, or -- if the quantity really varies -- bind it to a "
        "field that is traced."
    )


def deliver(
    value: Any,
    spec: FieldSpec,
    *,
    dtype: str,
    source: str = "scalar",
    declared_as: str | None = None,
    destination: DestinationDescriptor | None = None,
) -> Any:
    """Public compatibility wrapper for destination-aware delivery."""
    return deliver_checked(
        value, spec, dtype=dtype, source=source, declared_as=declared_as,
        destination=destination,
    )


def origin_for_delivery(
    context: ResolutionContext, destination: DestinationDescriptor, *, defaulted: bool = False
) -> Origin:
    """Find the payload authority exactly; never fabricate a user origin."""
    if defaulted:
        return Origin("rheplicant-default")
    if context.origin_lookup is None:
        raise ConfigError(f"audit: no origin lookup for {destination.document_path!r}")
    origin = context.origin_lookup(destination.document_path)
    if origin is None:
        raise ConfigError(f"audit: no origin for {destination.document_path!r}")
    return origin


_CANONICAL_AUDIT_UNITS = (
    "Hz", "s", "unix_s", "K", "deg", "m", "ohm", "dimensionless",
    "count", "samples", "bits", "channels", "cycles", "adc_count",
    "Hz/s", "adc_count/K", "dimensionless/s", "cycles/samples",
)
_EXPECTED_FROM_DESTINATION = object()


def canonical_unit_for_delivery(
    context: ResolutionContext,
    destination: DestinationDescriptor,
    explicit: Unit | None,
    *,
    expected: DimensionSignature | None | object = _EXPECTED_FROM_DESTINATION,
) -> str | None:
    """The canonical unit a destination receives, including implicit A9 units."""
    if explicit is not None:
        return explicit.canonical
    signature = (
        dimension_for(destination, context.dimensions)
        if expected is _EXPECTED_FROM_DESTINATION
        else expected
    )
    if signature is None:
        return None
    for token in _CANONICAL_AUDIT_UNITS:
        unit = canonical_unit(token)
        if dimension_of(unit) == signature:
            return unit.canonical
    raise ConfigError(
        f"audit: no canonical unit represents {destination.document_path!r}"
    )


def record_resolved_delivery(
    context: ResolutionContext,
    destination: DestinationDescriptor,
    unit: Unit | None,
    *,
    defaulted: bool = False,
    expected: DimensionSignature | None | object = _EXPECTED_FROM_DESTINATION,
) -> None:
    """Record a non-model value at the point its typed owner accepts it."""
    if context.trace is None:
        return
    context.trace.record_delivery(
        context.layer,
        destination,
        dtype=context.dtype,
        origin=origin_for_delivery(context, destination, defaulted=defaulted),
        unit=canonical_unit_for_delivery(
            context, destination, unit, expected=expected
        ),
    )


def deliver_checked(
    value: Any,
    spec: FieldSpec,
    *,
    dtype: str,
    source: str = "scalar",
    declared_as: str | None = None,
    destination: DestinationDescriptor | None = None,
) -> Any:
    """Coerce a resolved value into what ``spec``'s field will accept.

    Args:
        value: the resolved, canonical-unit value.
        spec: the destination, from :func:`field_specs`.
        dtype: the run's floating dtype, ``"float32"`` or ``"float64"``.
        source: the value form's name, for check A40 and for the message.
        declared_as: the document's own ``as:`` claim, cross-checked.

    Raises:
        ConfigError: on an array form landing on a static field (A40), on a
            declared ``as:`` the field contradicts, and on any value the
            destination's type cannot hold.
    """
    mode = mode_of(spec)
    if declared_as is not None:
        if declared_as not in DELIVERY_MODES:
            raise ConfigError(
                f"as={declared_as!r} is not a delivery mode; they are "
                f"{list(DELIVERY_MODES)}. The mode is normally inferred from the "
                "destination field's own metadata and only needs writing when you "
                "want the expectation checked."
            )
        if declared_as != mode:
            raise ConfigError(
                f"This value declares as={declared_as!r}, but field {spec.name!r} is "
                f"{mode!r} -- that is what its own eqx.field(...) metadata says, and "
                "the metadata is what equinox acts on. One of the two is out of date. "
                "Drop the as: key to take the field's word for it, or write the value "
                "against the field you meant."
            )
    if mode != "traced" and source in ARRAY_FORMS:
        _refuse_array_form(spec, source)

    if mode == "static_union":
        raise ConfigError(
            f"Field {spec.name!r} is declared {spec.annotation!r} -- a static "
            "field with more than one non-None type. This layer has no rule for "
            "choosing between them, and guessing is how a value arrives finite, "
            "correctly shaped and wrong. Give the field one type."
        )
    if mode != "traced":
        _, optional = _optional_payload(spec.annotation)
        if optional and value is None:
            # The annotation says ``T | None``; None is the declared way to
            # mean "leave this at its default", so it is delivered as itself
            # rather than pushed through T's guard.
            return None

    if mode == "traced":
        return _as_traced(value, spec, dtype)
    if mode == "static_bool":
        return _as_static_bool(value, spec)
    if mode == "static_int":
        return _as_static_int(value, spec)
    if mode == "static_float":
        return _as_static_float(value, spec)
    if mode == "static_str":
        return _as_static_str(value, spec)
    if mode == "static_tuple":
        return _as_tuple(value)
    if mode == "static_mapping":
        return FrozenMapping(value)
    # static_other: a Callable, a nested Module, a PyTreeDef. The value grammar
    # cannot make one of those; only the python: hatch can, and it passes the
    # object through untouched.
    return value


def _x64_enabled() -> bool:
    """Whether this process can represent float64, read at CALL time.

    Not at import: ``jax_enable_x64`` is normally switched on during a run's
    setup, which happens after this module has been imported. A snapshot taken
    at import would answer ``False`` for a run that had correctly enabled it.
    """
    try:
        return bool(jax.config.jax_enable_x64)
    except AttributeError:  # pragma: no cover - accessor differs by jax version
        return bool(jax.config.read("jax_enable_x64"))


def _require_x64(dtype: str) -> None:
    """Refuse a float64 request a float32 process would silently downcast.

    One guard covers ``complex128`` too, because that dtype is reachable only
    through ``dtype == "float64"``.
    """
    if dtype != "float64" or _x64_enabled():
        return
    raise ConfigError(
        "This value is to be delivered as float64, but jax_enable_x64 is False in "
        "this process, so jnp.asarray(...).astype('float64') silently returns "
        "float32 (and complex128 returns complex64). Nothing downstream can detect "
        "that: the arrays are finite, correctly shaped and plausibly structured, "
        "every later dtype check compares the downcast values against each other "
        "and agrees, and config.resolved.yaml would record float64 as though it had "
        "happened. The map<->alm steps shared by GeneralPointingProjector and "
        "DriftScanProjector (s2fft healpix transforms, Price-McEwen recursion) carry "
        "O(10%) errors in float32 even at small lmax, so the maps come out wrong by "
        "about a tenth rather than obviously broken. jax_enable_x64 is process-global "
        "and part of the hashed config, so it has to be set before any array exists: "
        "write runtime.jax_enable_x64: true in the document, or export "
        "JAX_ENABLE_X64=1 in the environment."
    )


def _as_traced(value: Any, spec: FieldSpec, dtype: str):
    _require_x64(dtype)
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        return array.astype("complex128" if dtype == "float64" else "complex64")
    if jnp.issubdtype(array.dtype, jnp.bool_):
        return array
    # Everything else becomes floating. An integer array here is not an
    # *inexact* array, so eqx.partition(op, eqx.is_inexact_array) drops it and
    # the field is silently uninferrable -- measured on AntennaLossOperator.
    return array.astype(dtype)


def _reject_numpy(value: Any, spec: FieldSpec) -> None:
    if type(value).__module__.startswith("numpy"):
        raise ConfigError(
            f"Field {spec.name!r} is static and the value is a numpy scalar "
            f"({type(value).__name__}). numpy.generic is one of equinox's array "
            "types, so it trips the 'A JAX array is being set as static!' warning, "
            "and isinstance(np.int64(8), int) is False so the operator's own guard "
            "refuses it too. Deliver a Python scalar."
        )


def _as_static_bool(value: Any, spec: FieldSpec) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(
            f"Field {spec.name!r} is a static bool and the value is "
            f"{type(value).__name__} ({value!r}). Write true or false."
        )
    return value


def _as_static_int(value: Any, spec: FieldSpec) -> int:
    if isinstance(value, bool):
        raise ConfigError(
            f"Field {spec.name!r} is a static int and the value is the bool {value!r}. "
            "Python's isinstance(True, int) is True, so this passes the operator's own "
            f"guard and gives {spec.name} = {int(value)} -- a one-bit ADC, or a "
            "single-sample period. Write the integer."
        )
    _reject_numpy(value, spec)
    if not isinstance(value, int):
        raise ConfigError(
            f"Field {spec.name!r} is a static int and the value is "
            f"{type(value).__name__} ({value!r}). The operator's own guard refuses "
            "this too, but only after it has been constructed -- which, on a beam "
            "node, is after the CST directory has been read and analysed. Write an "
            "integer."
        )
    return int(value)


def _as_static_float(value: Any, spec: FieldSpec) -> float:
    if isinstance(value, bool):
        raise ConfigError(
            f"Field {spec.name!r} is a static float and the value is the bool "
            f"{value!r}. Write {float(value)!r} if that is what was meant."
        )
    _reject_numpy(value, spec)
    if not isinstance(value, numbers.Real):
        raise ConfigError(
            f"Field {spec.name!r} is a static float and the value is "
            f"{type(value).__name__} ({value!r})."
        )
    return float(value)


def _as_static_str(value: Any, spec: FieldSpec) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            f"Field {spec.name!r} is a static str and the value is "
            f"{type(value).__name__} ({value!r})."
        )
    return value


def _as_tuple(value: Any):
    if isinstance(value, (list, tuple)):
        return tuple(_as_tuple(item) for item in value)
    return value
