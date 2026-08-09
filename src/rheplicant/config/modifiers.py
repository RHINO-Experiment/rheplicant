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
        value = jnp.asarray(value).astype(modifiers["dtype"])
    if "part" in modifiers:
        value = _part(value, modifiers["part"])
    if "scale" in modifiers or "offset" in modifiers:
        value = value * float(modifiers.get("scale", 1.0)) + float(modifiers.get("offset", 0.0))
    if "normalize" in modifiers:
        value = _normalize(value, modifiers["normalize"])
    if modifiers.get("column"):
        value = _column(value, form)
    return value


def _part(value: Any, part: str):
    array = jnp.asarray(value)
    return {"re": jnp.real, "im": jnp.imag, "abs": jnp.abs, "angle": jnp.angle}[part](array)


def _normalize(value: Any, kind: str):
    if kind == "none":
        return value
    array = jnp.asarray(value)
    if kind == "mean1":
        return array / jnp.mean(array)
    if kind == "pixel_sum":
        return array / jnp.sum(array)
    return array / jnp.max(array)


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
