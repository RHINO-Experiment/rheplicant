"""Form 3: a value drawn from a distribution, with its seed named in one place.

Three of the five stress-ported example scripts were blocked without this
(``driftscan_mmode.py:61``, ``sky_to_noise_wave.py:109``,
``three_ways_to_a_posterior.py:76``): their skies are *drawn*, not read, and
the alternative -- ship a binary blob whose provenance the config cannot state
-- defeats the whole argument for a resolved-config artefact.

A draw is a *generator*, not a computation: it has no operands to compose and
no result to feed back, so it does not open the door §2.3 closes.

``seed:`` names an entry of ``runtime.seeds`` and may not be a literal. That
is what keeps every realisation in a run enumerated in one place, which is
what ``provenance.json`` records. A name ``runtime.seeds`` does not declare is
derived from the root seed by a **blake2s** digest, not by Python's ``hash``:
the built-in string hash is salted per process, so v0's fallback gave a
different sky on every run of the same file.
"""

import hashlib
from typing import Any

import jax
import jax.numpy as jnp

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.symbols import resolve_extent
from rheplicant.config.units import convert_to_canonical
from rheplicant.config.values import ResolvedValue, register_form

_SEED_PREFIX = "runtime.seeds."


def _digest(name: str) -> int:
    """A stable 32-bit integer for a seed name.

    blake2s and not ``hash(name)``: Python's string hash is salted per process,
    so v0's ``fold_in(key, hash(name))`` drew a different sky on every run of
    the same file. Nothing reported it, because a different sky is still a
    plausible sky.
    """
    return int.from_bytes(hashlib.blake2s(name.encode("utf-8"), digest_size=4).digest())


def seed_for(name: str, context: ResolutionContext) -> int:
    """The reportable integer seed for ``name``, declared or derived.

    This is the number that lands in ``provenance.json``, and it is also the
    number the draw is made from -- :func:`_key` is defined in terms of this
    function rather than deriving a key of its own. See there.

    Raises:
        ConfigError: when the run declares no root seed at all -- ``seed:
            null`` is a legal, recorded state (``examples/sky_to_noise_wave.py``
            builds its State with no key), and it is incompatible with drawing.
    """
    if name in context.seeds:
        return int(context.seeds[name])
    if context.seed is None:
        raise ConfigError(
            f"A value node draws from a distribution with seed {name!r}, but this run "
            "declares runtime.seed: null and runtime.seeds does not name it. seed: "
            "null is a legal and recorded state -- it means State.key is None, and "
            "with no key SelectOperator does not split a subkey per branch, so a run "
            "that acquires one traverses a different PRNG path. Declare "
            f"runtime.seeds.{name}, or set runtime.seed and let this one be derived "
            "from it."
        )
    return _digest(name) ^ int(context.seed)


def _key(name: str, context: ResolutionContext):
    """The typed PRNG key a draw consumes, built from the reported seed.

    One function calling the other, rather than two derivations that happen to
    agree. They did not agree when this was written the obvious way: the
    reported seed was ``_digest(name) ^ root`` and the key was
    ``fold_in(key(root), _digest(name))``, which is a different key, so
    ``provenance.json`` named an integer that reproduced a different sky.
    Measured on the derived branch -- the run drew
    ``[0.262, 0.053, 0.581, ...]`` and reported the seed for
    ``[0.700, 0.891, 0.481, ...]``.

    Nothing downstream can catch that. Both arrays are finite, correctly
    shaped, and drawn from the right distribution; a different sky is still a
    plausible sky, and the only way to notice is to rerun from the manifest and
    diff, which is the thing the manifest exists to make unnecessary. So the
    two are not kept in step by a test -- a test can only check the pairs it
    thinks of -- they are the same expression.
    """
    return jax.random.key(seed_for(name, context))


def _seed_name(spec: dict, form: str) -> str:
    if "seed" not in spec:
        raise ConfigError(
            f"{form}: 'seed' is required and has no default. Every realisation in a "
            "run is enumerated in runtime.seeds so that provenance.json can record "
            "them, and a draw with no named seed is a number nothing can reproduce."
        )
    seed = spec["seed"]
    if not isinstance(seed, dict) or set(seed) != {"from"} or not isinstance(seed["from"], str):
        raise ConfigError(
            f"{form}: seed must NAME an entry of runtime.seeds -- "
            "{from: runtime.seeds.<name>} -- and got "
            f"{seed!r}. A literal here is a realisation that appears in one value node "
            "and nowhere else, so provenance.json cannot record it and a second run of "
            "the same file cannot be shown to be the same run."
        )
    target = seed["from"]
    if not target.startswith(_SEED_PREFIX) or not target[len(_SEED_PREFIX) :]:
        raise ConfigError(
            f"{form}: seed names {target!r}; it must be under {_SEED_PREFIX}<name>. "
            "runtime.seeds is an open namespace -- any name you like -- but it is the "
            "only namespace, because one place to look is the point of it."
        )
    return target[len(_SEED_PREFIX) :]


def _resolve_operand(node: Any, context: ResolutionContext, default: float) -> Any:
    from rheplicant.config.values import resolve_value

    if node is None:
        return default
    if isinstance(node, (int, float)):
        return float(node)
    return resolve_value(node, context).value


def _draw(form: str, node: dict, context: ResolutionContext, modifiers: dict) -> ResolvedValue:
    spec = node[form]
    if not isinstance(spec, dict):
        raise ConfigError(f"{form}: expects a mapping, got {type(spec).__name__} ({spec!r}).")
    known = {"shape", "seed"} | ({"loc", "scale"} if form == "normal" else {"low", "high"})
    unknown = sorted(set(spec) - known)
    if unknown:
        raise ConfigError(f"{form}: unknown key(s) {unknown}; its keys are {sorted(known)}.")
    if "shape" not in spec:
        raise ConfigError(f"{form}: 'shape' is required.")
    key = _key(_seed_name(spec, form), context)
    shape = tuple(resolve_extent(entry, context.shape_scope) for entry in spec["shape"])
    if form == "normal":
        loc = _resolve_operand(spec.get("loc"), context, 0.0)
        scale = _resolve_operand(spec.get("scale"), context, 1.0)
        array = loc + scale * jax.random.normal(key, shape, dtype=context.dtype)
    else:
        low = _resolve_operand(spec.get("low"), context, 0.0)
        high = _resolve_operand(spec.get("high"), context, 1.0)
        array = jax.random.uniform(key, shape, dtype=context.dtype, minval=low, maxval=high)
    unit_token = modifiers.get("unit")
    if unit_token is None:
        return ResolvedValue(array, None, form, modifiers)
    converted, unit = convert_to_canonical(jnp.asarray(array), unit_token)
    return ResolvedValue(converted, unit, form, modifiers)


@register_form("normal")
def _normal(node, context, modifiers):
    return _draw("normal", node, context, modifiers)


@register_form("uniform")
def _uniform(node, context, modifiers):
    return _draw("uniform", node, context, modifiers)
