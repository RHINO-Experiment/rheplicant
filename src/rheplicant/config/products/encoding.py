"""Deterministic, pickle-free encodings for scientific products."""

from __future__ import annotations

import io
import os
import zipfile
from collections.abc import Mapping

import numpy as np

from _rheplicant_bootstrap.audit.json import canonical_json_bytes
from _rheplicant_bootstrap.errors import ConfigError


def canonical_product_json(value: object) -> bytes:
    """Return canonical finite JSON bytes using the audit-layer contract."""
    return canonical_json_bytes(value)


def validate_relative_product_path(path: str, *, component_limit: int) -> None:
    """Validate one transaction-relative product path."""
    if type(path) is not str or not path or "\0" in path or "\\" in path:
        raise ConfigError("product path must be a non-empty portable relative path.")
    try:
        path.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError("product path must contain valid UTF-8.") from None
    if os.path.isabs(path):
        raise ConfigError("product path must be relative.")
    components = path.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise ConfigError("product path contains an ambiguous component.")
    if type(component_limit) is not int or component_limit <= 0:
        raise ConfigError("product path component limit must be a positive integer.")
    for component in components:
        size = len(os.fsencode(component))
        if size > component_limit:
            raise ConfigError(
                f"product path component {component!r} is {size} bytes; "
                f"filesystem limit is {component_limit}."
            )


def _array_key(key: object) -> str:
    if type(key) is not str or not key or "\0" in key or "\\" in key:
        raise ConfigError("NPZ keys must be non-empty portable strings.")
    if key.endswith(".npy"):
        raise ConfigError("NPZ keys must not include the .npy suffix.")
    components = key.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise ConfigError(f"NPZ key {key!r} contains an ambiguous component.")
    try:
        key.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError("NPZ keys must contain valid UTF-8.") from None
    return key


def deterministic_npz(
    values: Mapping[str, object],
) -> tuple[bytes, dict[str, dict[str, object]]]:
    """Encode sorted numeric arrays as a byte-stable NPZ archive."""
    if not isinstance(values, Mapping) or not values:
        raise ConfigError("NPZ product must contain at least one named array.")
    arrays: dict[str, np.ndarray] = {}
    for raw_key, value in values.items():
        key = _array_key(raw_key)
        if key in arrays:
            raise ConfigError(f"duplicate NPZ key {key!r}.")
        try:
            array = np.asarray(value)
        except Exception:
            raise ConfigError(f"NPZ value {key!r} cannot be converted to an array.") from None
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            raise ConfigError(f"NPZ value {key!r} must have a numeric non-object dtype.")
        arrays[key] = np.ascontiguousarray(array)

    archive_buffer = io.BytesIO()
    metadata: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            array = arrays[key]
            member_buffer = io.BytesIO()
            np.lib.format.write_array(member_buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, member_buffer.getvalue())
            metadata[key] = {"dtype": str(array.dtype), "shape": list(array.shape)}
    return archive_buffer.getvalue(), metadata


__all__ = [
    "canonical_product_json",
    "deterministic_npz",
    "validate_relative_product_path",
]
