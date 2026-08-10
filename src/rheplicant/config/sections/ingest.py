"""``observation.from_file`` (schema §4.1.2): the recording comes off disk.

``rhino_hdf5`` registers as an OBJECT reader (``array=False``): the return
value is a :class:`rheplicant.radio.rhino.RhinoObservation`, so the ``file:``
machinery must not ``jnp.asarray`` it and must refuse modifiers -- and in
exchange the path resolution and the sha256 digest come from ``files.py`` for
free. ``to_state`` does NOT run here: its ``source_order`` is read off the
assembled twin (``rhino.py:607-611``), so :mod:`rheplicant.config.document`
finishes the State after ``model:`` is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.files import register_reader
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.values import resolve_value

__all__ = ["parse_from_file"]

_FROM_FILE_KEYS = frozenset(
    {"format", "path", "sha256", "freq_unit", "thermistor_columns",
     "settle_seconds", "thermistor_unit"}
)


def _seconds(value: Any, key: str) -> float:
    """A plain number of seconds, or ``{value: ..., unit: s}`` -- the reader
    has no ResolutionContext, so the general value grammar stops here."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {"value", "unit"})
        if unknown or value.get("unit", "s") != "s" or "value" not in value:
            raise ConfigError(
                f"from_file: {key} is a number of seconds (or "
                f"{{value: ..., unit: s}}); got {dict(value)!r}."
            )
        value = value["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"from_file: {key} is a number of seconds; got "
            f"{type(value).__name__} ({value!r})."
        )
    return float(value)


@register_reader(
    "rhino_hdf5",
    frozenset({"freq_unit", "thermistor_columns", "settle_seconds",
               "thermistor_unit"}),
    array=False,
)
def _read_rhino_hdf5(path, spec: dict):
    """``radio/rhino.py:469`` ``read_rhino_observation``, into the file table.

    ``array=False``: the return value is a ``RhinoObservation`` -- the whole
    recording with its diagnostics -- not an array.
    """
    from rheplicant.radio.rhino import read_rhino_observation

    if "freq_unit" not in spec:
        raise ConfigError(
            "from_file: freq_unit is required and has no default -- the file "
            "does not record its frequency unit and its two producers "
            "disagree (rhino-cal writes Hz, the notebook writes MHz; "
            "radio/rhino.py's module docstring is the evidence)."
        )
    kwargs: dict[str, Any] = {"freq_unit": str(spec["freq_unit"])}
    if "thermistor_columns" in spec:
        columns = spec["thermistor_columns"]
        if (not isinstance(columns, Mapping)
                or not all(isinstance(k, str) for k in columns)
                or any(isinstance(v, bool) or not isinstance(v, int)
                       for v in columns.values())):
            raise ConfigError(
                "from_file: thermistor_columns is a mapping of switch label "
                f"-> integer column of /temperatures; got {columns!r}."
            )
        kwargs["thermistor_columns"] = dict(columns)
    if "settle_seconds" in spec:
        kwargs["settle_seconds"] = _seconds(spec["settle_seconds"],
                                            "settle_seconds")
    if "thermistor_unit" in spec:
        unit = spec["thermistor_unit"]
        if unit not in ("celsius", "kelvin"):
            raise ConfigError(
                f"from_file: thermistor_unit is 'celsius' (the file's "
                f"convention) or 'kelvin'; got {unit!r}."
            )
        kwargs["thermistor_unit"] = unit
    return read_rhino_observation(path, **kwargs)


def parse_from_file(spec: Any, context: ResolutionContext):
    """``(RhinoObservation, provenance_record)`` for an ingested observation."""
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"observation.from_file: is a mapping; got {type(spec).__name__}."
        )
    check_unknown_keys("observation.from_file", dict(spec), _FROM_FILE_KEYS,
                       label="from_file:")
    fmt = spec.get("format")
    if fmt != "rhino_hdf5":
        raise ConfigError(
            f"observation.from_file: format is 'rhino_hdf5' (the one "
            f"ingestion format this layer reads); got {fmt!r}."
        )
    if "path" not in spec:
        raise ConfigError("observation.from_file: requires path:.")
    resolved = resolve_value({"file": dict(spec)}, context)
    record = {
        "from_file/path": resolved.modifiers["_path"],
        "from_file/sha256": resolved.modifiers["_sha256"],
    }
    return resolved.value, record
