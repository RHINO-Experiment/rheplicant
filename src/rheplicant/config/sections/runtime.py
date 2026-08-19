"""``runtime:`` -> facts (schema §4.0).

Recorded and checked here; APPLIED by Plan 4's CLI. ``jax_enable_x64`` and
``platform`` are process-global -- ``jax.config.update`` must run before any
array exists -- so a library loader cannot honestly apply them mid-process.
What it can do is refuse the one configuration that would otherwise fail
later and worse: x64 physics declared in a float32 process surfaces here,
naming the fix, rather than at the first traced delivery.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import NamedTuple

import jax

from _rheplicant_bootstrap.process import parse_runtime
from _rheplicant_bootstrap.runtime import PriorEnvironment, RuntimeSession
from rheplicant.config.context import current_resolution_audit

__all__ = ["RuntimeFacts", "build_runtime", "state_key"]


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
    spec = parse_runtime(section)
    audit = current_resolution_audit()
    if audit is not None:
        if "jax_enable_x64" not in section:
            audit.use_default("runtime.jax_enable_x64", False)
        if "platform" not in section:
            audit.use_default("runtime.platform", "auto")
        if "seed" not in section:
            audit.use_default("runtime.seed", None)
        if "seeds" not in section:
            audit.use_default("runtime.seeds", {})
        audit.seed(spec.seed, spec.seeds)
    RuntimeSession(
        spec,
        PriorEnvironment(
            os.environ.get("JAX_ENABLE_X64"),
            os.environ.get("JAX_PLATFORMS"),
        ),
    ).verify(boundary="current process")
    return RuntimeFacts(
        jax_enable_x64=spec.jax_enable_x64,
        platform=spec.platform,
        seed=spec.seed,
        seeds=dict(spec.seeds),
    )


def state_key(facts: RuntimeFacts) -> jax.Array | None:
    """``State.key`` for this run: a typed PRNG key, or None for no seed."""
    if facts.seed is None:
        return None
    return jax.random.key(facts.seed)
