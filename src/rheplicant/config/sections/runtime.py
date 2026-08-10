"""``runtime:`` -> facts (schema §4.0).

Recorded and checked here; APPLIED by Plan 4's CLI. ``jax_enable_x64`` and
``platform`` are process-global -- ``jax.config.update`` must run before any
array exists -- so a library loader cannot honestly apply them mid-process.
What it can do is refuse the one configuration that would otherwise fail
later and worse: x64 physics declared in a float32 process surfaces here,
naming the fix, rather than at the first traced delivery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax

from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys

__all__ = ["RuntimeFacts", "build_runtime", "state_key"]

_RUNTIME_KEYS = frozenset({"jax_enable_x64", "platform", "seed", "seeds"})
_PLATFORMS = ("auto", "cpu", "gpu", "tpu")


class RuntimeFacts(NamedTuple):
    """What the ``runtime:`` section declared, checked and normalised."""

    jax_enable_x64: bool
    platform: str
    seed: int | None
    seeds: dict[str, int]

    @property
    def dtype(self) -> str:
        """The dtype every traced delivery in this run uses."""
        return "float64" if self.jax_enable_x64 else "float32"


def build_runtime(section: Mapping) -> RuntimeFacts:
    """Check and normalise the ``runtime:`` section."""
    if not isinstance(section, Mapping):
        raise ConfigError(
            f"runtime: is a mapping; got {type(section).__name__} ({section!r})."
        )
    check_unknown_keys(
        "runtime", dict(section), _RUNTIME_KEYS, label="the runtime section",
        hints={
            "x64_required_by": (
                "x64_required_by is emitted by the loader into the resolved "
                "record, never written by hand."
            )
        },
    )
    x64 = section.get("jax_enable_x64", False)
    if not isinstance(x64, bool):
        raise ConfigError(
            f"runtime.jax_enable_x64 is a bool; got {type(x64).__name__} ({x64!r})."
        )
    if x64 and not jax.config.jax_enable_x64:
        raise ConfigError(
            "runtime declares jax_enable_x64: true but this process is running "
            "float32. The flag is process-global and must be set before any "
            "array exists: jax.config.update('jax_enable_x64', True) at the top "
            "of your session (Plan 4's CLI applies it from the document "
            "automatically). Refused here rather than at the first traced "
            "delivery, where the message could not name the section."
        )
    platform = section.get("platform", "auto")
    if platform not in _PLATFORMS:
        raise ConfigError(
            f"runtime.platform is one of {list(_PLATFORMS)}; got {platform!r}."
        )
    seed = section.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ConfigError(
            f"runtime.seed is an int or null; got {type(seed).__name__} "
            f"({seed!r}). null is legal and recorded -- it means State.key = "
            "None, and a run that realises randomness will say so when it "
            "asks for the key."
        )
    seeds_spec = section.get("seeds", {})
    if not isinstance(seeds_spec, Mapping):
        raise ConfigError(
            f"runtime.seeds is a mapping of name -> int; got "
            f"{type(seeds_spec).__name__} ({seeds_spec!r})."
        )
    seeds: dict[str, int] = {}
    for name, value in seeds_spec.items():
        if not isinstance(name, str):
            raise ConfigError(f"runtime.seeds keys are strings; got {name!r}.")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(
                f"runtime.seeds.{name} is an int; got {type(value).__name__} "
                f"({value!r})."
            )
        seeds[name] = value
    return RuntimeFacts(
        jax_enable_x64=x64, platform=str(platform), seed=seed, seeds=seeds
    )


def state_key(facts: RuntimeFacts) -> jax.Array | None:
    """``State.key`` for this run: a typed PRNG key, or None for no seed."""
    if facts.seed is None:
        return None
    return jax.random.key(facts.seed)
