"""Canonical JSON bytes for audit envelopes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze_evidence, static_isinstance, thaw


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 JSON with one trailing newline."""
    frozen = freeze_evidence(value, where="audit JSON")
    pending = [frozen]
    while pending:
        item = pending.pop()
        if item is None or type(item) in (bool, int, str):
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ConfigError("audit JSON contains a non-finite number.")
            continue
        if static_isinstance(item, Mapping):
            pending.extend(item.values())
            continue
        if type(item) is tuple:
            pending.extend(item)
            continue
        raise ConfigError("audit JSON contains a value that is not JSON.")
    try:
        text = json.dumps(
            thaw(frozen),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        raise ConfigError("audit JSON serialization failed.") from None
    return (text + "\n").encode("utf-8")


__all__ = ["canonical_json_bytes"]
