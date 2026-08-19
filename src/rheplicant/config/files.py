"""Form 4: a file reference, its search path, its reader and its hash.

The reader table is a registry rather than an ``if`` chain for one reason: a
registry means the refusal for an unknown format lists what is actually
available today instead of a set someone remembered to update. That is the
shape ``core/graph.py:350`` ``register_graph`` established.

Plan 1B registers exactly one new format here, ``touchstone`` -- an object
reader (``array=False``: it returns a
:class:`~rheplicant.radio.touchstone.Touchstone`, not an array; see
:func:`register_reader`). Two more formats the schema names are real but are
not read through a value node at all: ``cst_dir`` and ``healpix`` build a
beam's raw array at ``resources.beams`` instead, because the frequency grid,
``nside`` and (for ``cst_dir``) ``phi0_deg``/``phi_sense`` are all in scope
there in a way a bare value node cannot express -- ``_ELSEWHERE`` below names
the route for both, so the refusal points somewhere real instead of claiming
the capability is absent. ``eqx_leaves`` arrives at ``model.<node>.eqx_leaves``
(Plan 2A Task 10): it reconstructs operator state onto a template built from
the node's own declared fields. ``rhino_hdf5`` is registered by
``rheplicant.config.sections.ingest`` as an object reader.

Every file reference is hashed. The cost is one read of a file that is about
to be read anyway, and it is what lets ``config.resolved.yaml`` state which
bytes a run saw -- so a rerun that disagrees is detectable rather than merely
suspected.

**What a document is trusted to do.** This is the one place the value grammar
reaches outside itself, so the assumption is worth writing down rather than
leaving to be inferred from what the code does not check. Path resolution
applies no containment: ``~`` and ``${ENV}`` expand, an absolute path is taken
as written, and a relative one may climb out of the document's directory with
``..``. That is deliberate, and every alternative breaks a spelling the
package is actually used with -- ``~/data/beams/...`` on a workstation,
``${SCRATCH}/...`` on a cluster, ``../data/...`` in a repository whose configs
and data are siblings. The assumption underneath is that whoever wrote the
document is whoever is running the pipeline, in which case the document can
already do nothing its author could not do at a shell.

That assumption stops holding the moment a document arrives from somewhere
else: a shared root, a CI artefact, a collaborator's YAML. Then a ``file:``
entry naming ``~/.ssh/id_ed25519`` is read by this process, and what lands in
``config.resolved.yaml`` is its ``_path`` and its ``_sha256``. The digest is
the part that turns a read into a disclosure -- it confirms a guess about a
file's contents to anyone holding the artefact, and the artefact exists to be
shared. At that point an opt-in "every resolved path must be under
``base_dir`` or a declared root" belongs in :func:`resolve_file_path`,
recorded alongside the roots it was checked against. It is not written yet
because on by default it would refuse all three spellings above, and the
threat it answers is not the one this layer was designed under. The decision
is recorded here, next to the function it would change, rather than in an
issue nobody reading this file would find.
"""

import hashlib
import os
import pathlib
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import jax.numpy as jnp
import numpy as np

from _rheplicant_bootstrap.capture import (
    CaptureService,
    ManifestEnumerator,
    register_capture_route,
)
from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.registry import LiveNames
from rheplicant.config.units import convert_to_canonical
from rheplicant.config.values import ResolvedValue, register_form

#: format name -> (reader, the keys it accepts beyond path/format/unit/sha256,
#: whether its return value is an array).
_READERS: dict[str, tuple[Callable[..., Any], frozenset[str], bool]] = {}


def register_reader(name: str, extra_keys: frozenset[str] = frozenset(), *, array: bool = True):
    """Register a file reader under ``name``. Returns the function.

    Args:
        name: the format token, as written in ``format:``.
        extra_keys: the keys this format accepts beyond path/format/sha256.
        array: ``True`` (the default) for a reader whose return value is an
            array -- ``jnp.asarray``'d and eligible for ``unit:`` like every
            other format. ``False`` for a reader that returns some other
            object (:class:`rheplicant.radio.touchstone.Touchstone`, today):
            the object is handed back unwrapped, and the node it came from
            takes no modifiers, because a modifier describes what an array's
            numbers ARE and an object-valued reader returns something else
            entirely.
    """

    def _register(fn):
        _READERS[name] = (fn, extra_keys, array)
        register_capture_route(name, owner=f"value-reader:{name}")
        return fn

    return _register


#: Every registered format, live. Plan 1B adds to it by importing its module.
FILE_FORMATS = LiveNames(_READERS)


def regular_reader_names() -> tuple[str, ...]:
    return tuple(sorted(_READERS))


@contextmanager
def _capture_for(context: ResolutionContext):
    if context.capture is not None:
        yield context.capture
        return
    root = pathlib.Path(tempfile.mkdtemp(prefix="rheplicant-capture-"))
    service = CaptureService(root)
    try:
        yield service
    finally:
        service.close()


def consume_captured_file(
    source: pathlib.Path,
    context: ResolutionContext,
    *,
    destination: DestinationDescriptor,
    format: str,
    reader: Callable[[pathlib.Path], Any],
    declared_sha256: str | None = None,
):
    with _capture_for(context) as service:
        return service.consume_file(
            source,
            layer=context.layer,
            destination=destination,
            format=format,
            reader=reader,
            declared_sha256=declared_sha256,
        )


def consume_captured_directory(
    source: pathlib.Path,
    context: ResolutionContext,
    *,
    destination: DestinationDescriptor,
    format: str,
    enumerate_manifest: ManifestEnumerator,
    reader: Callable[[pathlib.Path], Any],
    declared_sha256: str | None = None,
):
    with _capture_for(context) as service:
        return service.consume_directory(
            source,
            layer=context.layer,
            destination=destination,
            format=format,
            enumerate_manifest=enumerate_manifest,
            reader=reader,
            declared_sha256=declared_sha256,
        )


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


# Neither np.load below passes ``allow_pickle``, deliberately. numpy's default
# has been False since 1.16.3 for exactly this reason: an object-dtype .npy is
# a serialised Python object graph, and reconstructing one runs whatever code
# it names -- so with that default flipped, reading a config document that
# references an untrusted .npy is executing that file, at config-load time,
# before a single operator is built. Spelling the argument out even as False
# would not strengthen the guarantee; it would advertise the one-character
# edit, which is what the next person to hit "Object arrays cannot be loaded
# when allow_pickle=False" will reach for. The refusal is pinned by a test
# instead -- TestWhenAReaderFails.test_a_serialised_object_array_is_refused --
# so it cannot be relaxed without deleting an assertion that says why not.
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


def _read(
    reader,
    path: pathlib.Path,
    spec: dict,
    fmt: str,
    *,
    as_array: bool = True,
    diagnostic_path: pathlib.Path | None = None,
):
    """Call one reader, and let nothing out of it without the document's context.

    One wrapper here rather than a ``try`` inside each reader, for the reason
    :func:`rheplicant.config.values.resolve_value` gives for its single
    modifier exit: a reader added in a later task cannot opt out of this by
    forgetting to write it, and Plan 1B adds four. ``jnp.asarray`` is inside
    the guard too -- an object array that numpy declined to reject would fail
    there instead, and a bare ``TypeError`` from jax is no more use to a reader
    of the document than a bare ``ValueError`` from numpy.

    ``as_array=False`` skips the ``jnp.asarray`` and hands the reader's return
    value back exactly as given -- for a reader registered with
    ``register_reader(..., array=False)``, whose return value is not an array
    at all (:class:`rheplicant.radio.touchstone.Touchstone`, today) and which
    ``jnp.asarray`` would either mangle or refuse outright.

    The catch is ``Exception`` and not a list of the types numpy documents,
    because that list is not part of numpy's contract and does not survive
    contact with a malformed file: one bad ``.npz`` gives ``BadZipFile``, a
    truncated one ``EOFError``, a ragged ``.txt`` ``ValueError``, an
    out-of-range ``column:`` ``IndexError``. Enumerating them is how a guard
    ends up matching one shape of a failure and reading every other shape as
    success. The exception's own type is named in the message, so a defect in
    a reader is still reported as one rather than disguised as a bad file.
    """
    try:
        result = reader(path, spec)
        return jnp.asarray(result) if as_array else result
    except ConfigError:
        # A reader's own refusal already names the document, the key and the
        # remedy. Re-wrapping it would bury that under advice about delimiters.
        raise
    except Exception as exc:
        shown = path if diagnostic_path is None else diagnostic_path
        raise ConfigError(
            f"{shown} could not be read as format {fmt!r}. The reader raised "
            f"{type(exc).__name__}: {exc}. That message is the library's: it knows the "
            "file, and nothing about the document, the value node, or the keys that "
            "decided how the file would be parsed -- and one document may reference "
            "dozens of files. Three things usually differ from what was declared: the "
            "delimiter (csv assumes ',' unless the entry says otherwise), skiprows (a "
            "header row nothing was told to skip is parsed as data and fails on its "
            "first non-numeric field), and the format itself (a csv read as txt, or an "
            "npy from a producer that writes something else). Check those against the "
            "file's first few lines rather than against its extension, which this "
            "layer does not consult."
        ) from exc


#: Formats that are real but are not read through a value node, and where they are.
_ELSEWHERE: dict[str, str] = {
    "cst_dir": "resources.beams, with format: cst -- it needs the frequency grid, an "
               "nside, and phi0_deg/phi_sense, none of which a value node can carry",
    "healpix": "resources.beams, with format: healpix -- it needs order: (RING versus "
               "NESTED is declared, not guessed), the declared frequency grid, and "
               "frame:, none of which a value node can carry",
}


def _refuse_healpix(spec: dict) -> None:
    raise ConfigError(
        "format: healpix is not read through a value node. It lives at "
        + _ELSEWHERE["healpix"]
        + ". A bare value node has nowhere to put those declarations, and a HEALPix "
        "map read under a guessed ordering keeps its shape and its statistics with "
        "every pixel in the wrong place."
    )


@register_form("file")
def _file(node, context, modifiers, target):
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
    # Before the table lookup, deliberately. None of _ELSEWHERE's formats are
    # registered, so the generic refusal below would answer them first and name
    # neither remedy -- "unknown format" reads as a typo, which invites the
    # reader to try another spelling of a reader that does not exist in any
    # spelling. healpix keeps its own wording (the ordering-guess hazard is
    # worth spelling out); cst_dir gets the route named plainly.
    if fmt == "healpix":
        _refuse_healpix(spec)
    elif fmt in _ELSEWHERE:
        raise ConfigError(
            f"file: format {fmt!r} is not read through a value node. It lives at "
            f"{_ELSEWHERE[fmt]}."
        )
    entry = _READERS.get(fmt)
    if entry is None:
        raise ConfigError(
            f"file: unknown format {fmt!r}; the registered readers are "
            f"{sorted(_READERS)}. Each names both a library call and the keys that "
            "call takes, so a format this layer does not know is refused rather than "
            "guessed from the extension."
        )
    reader, extra, is_array = entry
    unknown = sorted(set(spec) - {"path", "format", "sha256"} - extra)
    if unknown:
        raise ConfigError(
            f"file: format {fmt!r} does not take {unknown}; beyond path, format and "
            f"sha256 it takes {sorted(extra)}."
        )

    path = resolve_file_path(spec["path"], context)
    destination = (
        target.destination
        if target is not None
        else DestinationDescriptor("file", "config_path", "file")
    )

    def read_snapshot(snapshot: pathlib.Path):
        result = _read(
            reader,
            snapshot,
            spec,
            fmt,
            as_array=is_array,
            diagnostic_path=path,
        )
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        return result, digest

    result, digest = consume_captured_file(
        path,
        context,
        destination=destination,
        format=fmt,
        reader=read_snapshot,
        declared_sha256=spec.get("sha256"),
    )
    carried = {**modifiers, "_sha256": digest, "_path": str(path)}
    if not is_array:
        # Modifiers describe what an array's numbers ARE (unit, part, scale,
        # normalize, ...); an object-valued reader returns something else
        # entirely, and there is no modifier here that means anything applied
        # to it. Refused by name -- `type(result).__name__` -- rather than
        # dropped, on the same reasoning `resolve_value`'s own unknown-key
        # sweep gives for silently-ignored keys.
        if modifiers:
            raise ConfigError(
                "file: an object-valued file node takes no modifiers; modifiers "
                f"describe arrays, and this reader returns a {type(result).__name__}."
            )
        return ResolvedValue(result, None, "file", carried)

    unit_token = modifiers.get("unit")
    if unit_token is None:
        return ResolvedValue(result, None, "file", carried)
    converted, unit = convert_to_canonical(result, unit_token)
    return ResolvedValue(converted, unit, "file", carried)
