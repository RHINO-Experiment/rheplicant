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

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.files import register_reader
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.values import resolve_value

__all__ = ["freq_unit_problem", "parse_from_file"]

_FROM_FILE_KEYS = frozenset(
    {"format", "path", "sha256", "freq_unit", "thermistor_columns",
     "settle_seconds", "thermistor_unit"}
)

#: The one ingestion format whose reader demands ``freq_unit:``.  Bound here
#: rather than spelled twice: :func:`parse_from_file` refuses any other
#: ``observation.from_file`` format outright, and
#: ``config.preflight.ingest`` needs the same token to tell a ``{file: ...}``
#: value node that reaches THIS reader from one that does not.
RHINO_FORMAT = "rhino_hdf5"


def freq_unit_problem(spec: Any) -> str | None:
    """Check A10: a ``rhino_hdf5`` file node must declare ``freq_unit:``.

    The message, or ``None`` when the spec declares one.

    **One binding, two callers** (§2.2): :func:`_read_rhino_hdf5` below, and
    ``config.preflight.ingest._freq_unit``, which asks the same question of
    the document's raw text one phase earlier.  The words are unchanged --
    this is a hoist, and the sentence said the right thing in the wrong place.

    **What the wrong place costs, measured at ``ea4839b``.**  With
    ``hashlib.sha256`` and ``Path.read_bytes`` instrumented, loading a
    document whose ``from_file:`` omits ``freq_unit`` records
    ``['read_bytes', 'sha256']`` before this refusal is reached: the whole
    recording is slurped off disk and hashed to miss a one-token key.  The
    digest is not moved to fix that -- it is read TODAY and not only by a
    later plan, because :func:`parse_from_file` writes ``from_file/sha256``
    into the provenance record and
    ``tests/config/test_config_document.py`` asserts it is there.  What moves
    is the question, into a pass that runs before any file is opened.

    ``Any`` rather than ``Mapping``: the pre-flight caller hands over whatever
    the document holds at that key, and a non-mapping is
    :func:`parse_from_file`'s refusal (or the ``file:`` grammar's), not this
    one's.  Answering ``None`` for it is the stand-down, not an oversight.
    """
    if isinstance(spec, Mapping) and "freq_unit" in spec:
        return None
    return (
        "from_file: freq_unit is required and has no default -- the file "
        "does not record its frequency unit and its two producers "
        "disagree (rhino-cal writes Hz, the notebook writes MHz; "
        "radio/rhino.py's module docstring is the evidence)."
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
    RHINO_FORMAT,
    frozenset({"freq_unit", "thermistor_columns", "settle_seconds",
               "thermistor_unit"}),
    array=False,
)
def _read_rhino_hdf5(path, spec: dict):
    """``radio/rhino.py:469`` ``read_rhino_observation``, into the file table.

    ``array=False``: the return value is a ``RhinoObservation`` -- the whole
    recording with its diagnostics -- not an array.

    The ``freq_unit`` refusal stays here as the backstop (§2.2 step 3): the
    pre-flight pass reads the document's text and this reader is reached from
    routes that text alone cannot always see.
    """
    from rheplicant.radio.rhino import read_rhino_observation

    problem = freq_unit_problem(spec)
    if problem is not None:
        raise ConfigError(problem)
    kwargs: dict[str, Any] = {"freq_unit": str(spec["freq_unit"])}
    if "thermistor_columns" in spec:
        columns = spec["thermistor_columns"]
        if columns is not None and (not isinstance(columns, Mapping)
                or not all(isinstance(k, str) for k in columns)
                or any(isinstance(v, bool) or not isinstance(v, int)
                       for v in columns.values())):
            raise ConfigError(
                "from_file: thermistor_columns is a mapping of switch label "
                f"-> integer column of /temperatures; got {columns!r}."
            )
        kwargs["thermistor_columns"] = (
            None if columns is None else dict(columns)
        )
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
    declared = dict(spec)
    check_unknown_keys("observation.from_file", declared, _FROM_FILE_KEYS,
                       label="from_file:")
    fmt = declared.get("format")
    if fmt != RHINO_FORMAT:
        raise ConfigError(
            f"observation.from_file: format is 'rhino_hdf5' (the one "
            f"ingestion format this layer reads); got {fmt!r}."
        )
    if "path" not in declared:
        raise ConfigError("observation.from_file: requires path:.")
    if "thermistor_columns" not in declared:
        declared["thermistor_columns"] = context.use_default(
            "observation.from_file.thermistor_columns",
            None,
        )
    if "settle_seconds" not in declared:
        declared["settle_seconds"] = context.use_default(
            "observation.from_file.settle_seconds",
            5.0,
        )
    if "thermistor_unit" not in declared:
        declared["thermistor_unit"] = context.use_default(
            "observation.from_file.thermistor_unit",
            "celsius",
        )
    resolved = resolve_value(
        {"file": declared},
        context,
        destination=DestinationDescriptor(
            "observation.from_file", "config_path", "observation.from_file"
        ),
    )
    record = {
        "from_file/path": resolved.modifiers["_path"],
        "from_file/sha256": resolved.modifiers["_sha256"],
    }
    return resolved.value, record
