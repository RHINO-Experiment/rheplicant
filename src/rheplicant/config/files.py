"""Form 4: a file reference, its search path, its reader and its hash.

The reader table is a registry rather than an ``if`` chain for one reason:
Plan 1B adds four more formats that construct package objects
(``touchstone``, ``cst_dir``, ``rhino_hdf5``, ``eqx_leaves``), and a registry
means the refusal for an unknown format lists what is actually available today
instead of a set someone remembered to update. That is the shape
``core/graph.py:350`` ``register_graph`` established.

Every file reference is hashed. The cost is one read of a file that is about
to be read anyway, and it is what lets ``config.resolved.yaml`` state which
bytes a run saw -- so a rerun that disagrees is detectable rather than merely
suspected.
"""

import hashlib
import os
import pathlib
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.registry import LiveNames
from rheplicant.config.units import convert_to_canonical
from rheplicant.config.values import ResolvedValue, register_form

#: format name -> (reader, the keys it accepts beyond path/format/unit/sha256)
_READERS: dict[str, tuple[Callable[..., Any], frozenset[str]]] = {}


def register_reader(name: str, extra_keys: frozenset[str] = frozenset()):
    """Register a file reader under ``name``. Returns the function."""

    def _register(fn):
        _READERS[name] = (fn, extra_keys)
        return fn

    return _register


#: Every registered format, live. Plan 1B adds to it by importing its module.
FILE_FORMATS = LiveNames(_READERS)


def resolve_file_path(
    raw: str, context: ResolutionContext, *, must_exist: bool = True
) -> pathlib.Path:
    """Resolve a declared path: the document's directory, then roots, then absolute.

    ``~`` and ``${ENV}`` expand first. The order is the schema's, and the
    refusal names every place that was tried -- a path that resolved somewhere
    unexpected is the failure this ordering exists to make visible.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(raw)))
    candidate = pathlib.Path(expanded)
    tried: list[pathlib.Path] = []
    if candidate.is_absolute():
        tried.append(candidate)
    else:
        if context.base_dir is not None:
            tried.append(pathlib.Path(context.base_dir) / candidate)
        tried.extend(pathlib.Path(root) / candidate for root in context.roots)
        tried.append(candidate.resolve())
    if not must_exist:
        return tried[0]
    for path in tried:
        if path.exists():
            return path
    raise ConfigError(
        f"No file at {raw!r}. It was looked for, in order, at: "
        + "; ".join(str(path) for path in tried)
        + ". A relative path resolves against the directory of the document that "
        "mentions it, then against each entry of paths.roots in order, then as "
        "written -- so a file that moved is refused here rather than read from "
        "whichever copy happened to be on the search path."
    )


@register_reader("npy")
def _read_npy(path: pathlib.Path, spec: dict):
    return np.load(path)


@register_reader("npz", frozenset({"key"}))
def _read_npz(path: pathlib.Path, spec: dict):
    archive = np.load(path)
    key = spec.get("key")
    if key is None:
        raise ConfigError(
            f"{path.name} is an npz archive, which holds several arrays, and no 'key' "
            f"was named. Its keys are {list(archive.files)}. There is no default: "
            "picking the first would depend on the order numpy happened to write them."
        )
    if key not in archive.files:
        raise ConfigError(
            f"{path.name} has no array named {key!r}; it holds {list(archive.files)}."
        )
    return archive[key]


@register_reader("txt", frozenset({"column", "columns", "skiprows"}))
def _read_txt(path: pathlib.Path, spec: dict):
    data = np.loadtxt(path, skiprows=int(spec.get("skiprows", 0)))
    if "column" in spec:
        return data[:, int(spec["column"])]
    if "columns" in spec:
        return data[:, [int(index) for index in spec["columns"]]]
    return data


@register_reader("csv", frozenset({"columns", "delimiter"}))
def _read_csv(path: pathlib.Path, spec: dict):
    data = np.genfromtxt(
        path, delimiter=spec.get("delimiter", ","), names=True, dtype=None, encoding="utf-8"
    )
    columns = spec.get("columns")
    if columns is None:
        return np.stack([data[name] for name in data.dtype.names], axis=-1)
    missing = [name for name in columns if name not in (data.dtype.names or ())]
    if missing:
        raise ConfigError(
            f"{path.name} has no column(s) {missing}; its header names "
            f"{list(data.dtype.names or ())}."
        )
    if len(columns) == 1:
        return data[columns[0]]
    return np.stack([data[name] for name in columns], axis=-1)


def _refuse_healpix(spec: dict) -> None:
    raise ConfigError(
        "format: healpix is not available. There is no HEALPix map reader anywhere in "
        "this package, and adding one is a package change with its own decision to "
        "take: it needs healpy -- which is undeclared here and arrives transitively "
        "through limTOD -- and it must settle RING versus NESTED and which axis "
        "carries frequency, neither of which a FITS file states unambiguously. Two "
        "routes work today: save the maps as npy or npz and declare nside on the "
        "entry, which is checked against 12*nside**2; or build the sky through the "
        "python: hatch, which states its own cost."
    )


@register_form("file")
def _file(node, context, modifiers):
    spec = node["file"]
    if not isinstance(spec, dict):
        raise ConfigError(f"file: expects a mapping, got {type(spec).__name__} ({spec!r}).")
    for required in ("path", "format"):
        if required not in spec:
            raise ConfigError(
                f"file: {required!r} is required. A reference states both where the "
                "bytes are and how to read them; the extension is not consulted, "
                "because two producers of the same extension disagree often enough "
                "that guessing is how a run reads the wrong thing quietly."
            )
    fmt = spec["format"]
    # Before the table lookup, deliberately. healpix is not registered, so the
    # generic refusal below would answer it first and name neither remedy --
    # and "unknown format" reads as a typo, which invites the reader to try
    # another spelling of a reader that does not exist in any spelling.
    if fmt == "healpix":
        _refuse_healpix(spec)
    entry = _READERS.get(fmt)
    if entry is None:
        raise ConfigError(
            f"file: unknown format {fmt!r}; the registered readers are "
            f"{sorted(_READERS)}. Each names both a library call and the keys that "
            "call takes, so a format this layer does not know is refused rather than "
            "guessed from the extension."
        )
    reader, extra = entry
    unknown = sorted(set(spec) - {"path", "format", "sha256"} - extra)
    if unknown:
        raise ConfigError(
            f"file: format {fmt!r} does not take {unknown}; beyond path, format and "
            f"sha256 it takes {sorted(extra)}."
        )

    path = resolve_file_path(spec["path"], context)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    declared = spec.get("sha256")
    if declared is not None and declared != digest:
        raise ConfigError(
            f"{path} hashes to {digest}, and this reference declares {declared}. The "
            "file has changed since the declaration was written, or the declaration "
            "came from a different copy. A run against different bytes than the ones "
            "recorded is not the run the artefact describes."
        )

    array = jnp.asarray(reader(path, spec))
    unit_token = modifiers.get("unit")
    carried = {**modifiers, "_sha256": digest, "_path": str(path)}
    if unit_token is None:
        return ResolvedValue(array, None, "file", carried)
    converted, unit = convert_to_canonical(array, unit_token)
    return ResolvedValue(converted, unit, "file", carried)
