"""The eight modifiers: what a value IS, said alongside what it holds.

Each of these is a declaration the array itself cannot carry. ``normalize:``
is the clearest case: a beam's numbers do not say whether they are a unit-sum
map or raw CST gain, and the output's unit is decided by the *pair*
(beam normalisation, ``normalize_beam``) -- 32838 K against 200 K on a uniform
200 K sky. ``column:`` is the second: on a square grid ``(n,)`` and ``(n, 1)``
are indistinguishable by shape and mean per-frequency and per-sample.

``scale:``/``offset:`` is the boundary of what this grammar computes. One
level, no nesting, no precedence: ``scale * v + offset``. Anything past it --
trig, powers, a product of two value nodes -- goes to the ``python:`` hatch
and pays the stated cost.
"""

from typing import Any

import jax.numpy as jnp

from rheplicant.config.errors import ConfigError

#: `part:` -- which component of a complex value node.
PARTS: tuple[str, ...] = ("re", "im", "abs", "angle")

#: `normalize:` -- a declaration of convention, never a computation the array
#: could have implied.
NORMALIZATIONS: tuple[str, ...] = ("none", "mean1", "pixel_sum", "max1")

#: `dtype:` -- the four the run may ask for.
DTYPES: tuple[str, ...] = ("float32", "float64", "complex64", "complex128")

#: `axis:` -- mandatory for a 1-D noise sigma; recorded here, checked by
#: inference/noise.py:264 check_noise_std_axis.
NOISE_AXES: tuple[str, ...] = ("time", "freq", "none")

#: The members of :data:`DTYPES` that cannot hold a phase. Derived from the
#: table rather than written out, so the two stay in step.
REAL_DTYPES: tuple[str, ...] = tuple(name for name in DTYPES if not name.startswith("complex"))


def apply_modifiers(value: Any, modifiers: dict[str, Any], *, form: str) -> Any:
    """Apply every modifier that transforms the value, in a fixed order.

    Order: ``dtype`` -> ``part`` -> ``scale``/``offset`` -> ``normalize`` ->
    ``column``. It is fixed rather than written so that two documents with the
    same keys cannot mean two different things; each step is documented on the
    function that performs it.

    ``unit:`` is applied by the form itself (it decides what the number means,
    not what it looks like) and ``as:`` and ``axis:`` are recorded rather than
    applied -- ``as:`` is cross-checked against the destination in
    :mod:`rheplicant.config.delivery`, and ``axis:`` is read by the noise model.

    Every key is looked up by name and anything else in the mapping is passed
    over rather than refused. That is not laxity -- the dispatcher has already
    refused any key the document wrote that is not a modifier -- it is what
    lets a form record its own findings in the same dict: ``arrays._finish``
    writes ``_shadowed`` there for check A41.

    Args:
        value: the form's result, already in canonical units.
        modifiers: the node's modifier keys as written, plus whatever the form
            recorded alongside them.
        form: the form key that produced ``value``, quoted in refusals.

    Raises:
        ConfigError: on a ``part``, ``normalize``, ``dtype`` or ``axis`` outside
            its table, and on ``column: true`` over anything not 1-D.
    """
    for key, permitted in (
        ("part", PARTS),
        ("normalize", NORMALIZATIONS),
        ("dtype", DTYPES),
        ("axis", NOISE_AXES),
    ):
        if key in modifiers and modifiers[key] not in permitted:
            raise ConfigError(_unknown_modifier_message(key, modifiers[key], permitted))

    if "dtype" in modifiers:
        value = _cast(value, modifiers["dtype"], form)
    if "part" in modifiers:
        value = _part(value, modifiers["part"])
    if "scale" in modifiers or "offset" in modifiers:
        value = value * float(modifiers.get("scale", 1.0)) + float(modifiers.get("offset", 0.0))
    if "normalize" in modifiers:
        value = _normalize(value, modifiers["normalize"])
    if modifiers.get("column"):
        value = _column(value, form)
    return value


def _cast(value: Any, dtype: str, form: str):
    """Apply ``dtype:``, refusing the one direction of it that loses meaning.

    Widening a real value to complex is lossless and stays legal. Narrowing a
    complex one is not: it deletes the phase and leaves a real number of
    entirely plausible magnitude. jax itself is on the way to refusing it --
    ``astype`` already warns ``Casting from complex to real dtypes will soon
    raise a ValueError`` -- but a DeprecationWarning is not a refusal, and it
    names neither the document nor the key that asked for it.
    """
    array = jnp.asarray(value)
    if dtype in REAL_DTYPES and jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ConfigError(
            f"dtype: {dtype!r} is a real dtype and this {form} value is complex "
            f"({array.dtype}). jnp.asarray(...).astype({dtype!r}) does not raise -- "
            "measured, 1+2j comes back as 1.0 -- so the imaginary part is dropped and "
            "what is left is a real number of entirely plausible magnitude that is a "
            "different quantity. Every complex value in this package is a reflection "
            "coefficient (gamma_src, gamma_rec, the s_params resources) and the "
            "noise-wave receiver model is built on their phase, so NoiseWaveOperator "
            "consumes the stripped value without complaint and the answer comes back "
            "finite, correctly shaped and wrong. Taking a component is a declaration "
            f"rather than a cast, so replace dtype: {dtype} with part: re if the real "
            "part is what was meant, or with part: abs or part: angle if the magnitude "
            "or the phase was. Replace rather than add: the order here is dtype then "
            f"part, so writing both would still narrow first. The real dtypes are "
            f"{list(REAL_DTYPES)}; dtype: is for widening a real value, which loses "
            "nothing."
        )
    return array.astype(dtype)


def _part(value: Any, part: str):
    array = jnp.asarray(value)
    return {"re": jnp.real, "im": jnp.imag, "abs": jnp.abs, "angle": jnp.angle}[part](array)


def _normalize(value: Any, kind: str):
    if kind == "none":
        return value
    array = jnp.asarray(value)
    if kind == "mean1":
        return array / _usable_divisor(jnp.mean(array), kind, "mean")
    if kind == "pixel_sum":
        return array / _usable_divisor(jnp.sum(array), kind, "sum")
    return array / _usable_divisor(jnp.max(array), kind, "maximum")


def _usable_divisor(divisor: Any, kind: str, name: str):
    """Return ``divisor``, or refuse if dividing by it cannot mean anything.

    ``isfinite`` first and ``== 0`` second, as two explicit branches rather
    than one comparison. A nan divisor answers ``False`` to ``> 0``, to
    ``< 0`` and to ``== 0`` alike, so every ordering-based spelling of this
    guard admits exactly the worst of the three cases -- and it is the worst
    because a nan then defeats every comparison-based check downstream too.
    """
    if bool(jnp.isfinite(divisor)) and not bool(divisor == 0):
        return divisor
    raise ConfigError(
        f"normalize: {kind!r} divides by the value's {name}, and this value's {name} "
        f"is {divisor}, so every element would come back inf or nan. This is the last "
        "point at which that can be said: nan > x is False and nan != nan is True, so "
        "a guard written downstream as 'refuse anything outside the range' passes a "
        "nan straight through -- which has happened four separate times in this "
        "package. The two routes here are not equally visible either. pixel_sum on a "
        "beam whose pixels cancel hands the projector a nan beam and the TOD comes "
        "back nan with every shape correct; an all-zero mean1 bandpass is simply the "
        "shape a partially-written config has, {zeros: [n_freq]} with a normalize: "
        "still attached from the template. Write normalize: none if the value is meant "
        f"to be used as it stands, or fix the value so its {name} is a non-zero finite "
        f"number. The conventions are {list(NORMALIZATIONS)}."
    )


def _column(value: Any, form: str):
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ConfigError(
            f"column: true forces (n,) to (n, 1) and this {form} value has shape "
            f"{tuple(array.shape)}. (n, 1) is the only shape CalLoadOperator.t_load "
            "reads as per-sample; on a square grid it is the only thing that "
            "distinguishes per-sample from per-frequency, which is why the key is "
            "written rather than inferred. Drop column:, or produce a 1-D value."
        )
    return array[:, None]


def _unknown_modifier_message(key: str, given: Any, permitted: tuple[str, ...]) -> str:
    why = {
        "part": (
            "A complex value has four components and the array does not say which "
            "one the field wants."
        ),
        "normalize": (
            "This is a declaration of convention, not a computation: the "
            "output's unit is decided by the pair (beam normalisation, "
            "normalize_beam), measured at 32838 K against 200 K on a uniform "
            "200 K sky, and the numbers alone cannot say which half is which."
        ),
        "dtype": (
            "x64 is process-global and part of the hashed config; a dtype this "
            "layer does not name would be silently downcast."
        ),
        "axis": (
            "A 1-D sigma is per-time or per-frequency and the shape cannot say "
            "which: on a square grid the wrong choice gives error bars of "
            "0.00004..0.00354 against a flat 0.00010, with every shape correct. "
            "inference/noise.py:264 check_noise_std_axis is the guard this feeds."
        ),
    }[key]
    return (
        f"{key}={given!r} is not one of {list(permitted)}. {why} Write one of them, or "
        f"drop {key}: entirely if the value needs no such declaration."
    )
