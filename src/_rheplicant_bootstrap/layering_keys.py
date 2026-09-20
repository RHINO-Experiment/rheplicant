"""The one piece of layering grammar both halves of the merge need.

``~key`` deletes. :mod:`_rheplicant_bootstrap.layering` reads it while
merging origins and :mod:`_rheplicant_bootstrap.layering_copy` reads it while
merging the defensive copy, so it lives in neither -- a module this small
exists to keep those two from importing each other.
"""

from __future__ import annotations

from _rheplicant_bootstrap.errors import ConfigError


def _deletion_target(key: str) -> str:
    target = key[1:]
    if not target:
        raise ConfigError("layering deletion key must name a value after '~'.")
    return target
