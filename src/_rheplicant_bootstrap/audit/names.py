"""Injective filesystem-safe encodings for document names."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Literal

from _rheplicant_bootstrap.errors import ConfigError

from .types import PathEncoding


def encode_name(name: str) -> str:
    return "n-" + name.encode("utf-8").hex()


def validate_encoded_names(
    kind: Literal["run", "variant"],
    names: Iterable[str],
    *,
    component_limit: int,
) -> Sequence[PathEncoding]:
    if kind not in ("run", "variant"):
        raise ConfigError(f"unknown encoded-name kind {kind!r}.")
    raw_names = tuple(names)
    for name in raw_names:
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{kind} names must be non-empty strings.")
    rows = tuple(PathEncoding(kind, name, encode_name(name)) for name in raw_names)
    folded = [os.path.normcase(row.encoded_name).casefold() for row in rows]
    if len(folded) != len(set(folded)):
        raise ConfigError(f"encoded {kind} output names collide.")
    for row in rows:
        size = len(os.fsencode(row.encoded_name))
        if size > component_limit:
            raise ConfigError(
                f"encoded {kind} name for {row.document_name!r} is {size} "
                f"bytes; filesystem limit is {component_limit}."
            )
    return rows
