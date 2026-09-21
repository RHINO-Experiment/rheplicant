"""The output grammar: what a document may ASK for.

Which keys an `output:` block accepts, which product and report formats exist,
and the parse that turns the block into a request. Everything here is about
the document's text; nothing here touches the filesystem.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import static_isinstance

from .types import (
    ParsedOutputSection,
    ProductRequest,
    ReportRequest,
)

_OUTPUT_KEYS = ("dir", "clobber", "stdout", "write", "report")

_WRITE_KEYS = ("config", "provenance", "diagnostics")

_PLAN4B_WRITE = (
    "arrays",
    "aux",
    "taps",
    "assembly",
    "estimates",
    "parameters",
    "draws",
    "losses",
    "gradients",
    "covariance",
    "prediction_bands",
    "posterior_predictives",
    "identifiability",
    "scores",
    "recovery",
    "training_history",
    "timings",
    "refusals",
    "signal_paths",
    "compare",
    "benchmark",
    "chains",
    "run_diagnostics",
)

_PLAN4B_TOP = ("memory_archive", "posterior_net", "campaign")

_PRODUCT_DEFAULT_FORMATS = {
    "arrays": "npz",
    "aux": "npz",
    "taps": "npz",
    "assembly": "json",
    "estimates": "npz",
    "parameters": "npz",
    "draws": "npz",
    "losses": "npz",
    "gradients": "npz",
    "covariance": "npz",
    "prediction_bands": "npz",
    "posterior_predictives": "npz",
    "identifiability": "json",
    "scores": "npz",
    "recovery": "json",
    "training_history": "npz",
    "timings": "json",
    "refusals": "txt",
    "signal_paths": "svg",
    "compare": "json",
    "benchmark": "json",
    "chains": "npz",
    "run_diagnostics": "json",
}

_PRODUCT_FORMATS = {
    **{name: (format_,) for name, format_ in _PRODUCT_DEFAULT_FORMATS.items()},
    "signal_paths": ("svg", "html", "mermaid"),
    "chains": ("npz", "netcdf"),
}

_REPORT_COLUMNS = ("mean", "std", "seconds")

_REPORT_RELATIVE = ("mean_sigma", "width_ratio")

_REPORT_FORMATS = ("text", "json")

_STDOUT = ("none", "summary", "verbose")

_COMMANDS = ("validate", "run", "script")


def _mapping(value: object, *, where: str) -> Mapping[object, object]:
    if not static_isinstance(value, Mapping):
        raise ConfigError(f"{where}: must be a mapping.")
    return cast(Mapping[object, object], value)


def _closed_mapping(
    value: object,
    *,
    where: str,
    allowed: tuple[str, ...],
) -> dict[str, object]:
    raw = _mapping(value, where=where)
    result: dict[str, object] = {}
    try:
        pairs = tuple(raw.items())
    except Exception:
        raise ConfigError(f"{where}: mapping traversal failed.") from None
    for key, item in pairs:
        if not static_isinstance(key, str):
            raise ConfigError(f"{where}: keys must be strings.")
        name = str.__str__(key)
        if name in result:
            raise ConfigError(f"{where}: keys collide after canonicalization.")
        if name not in allowed:
            raise ConfigError(f"{where}.{name}: unknown key")
        result[name] = item
    return result


def _required_true(value: object, *, where: str) -> bool:
    if not static_isinstance(value, bool):
        raise ConfigError(f"{where}: must be true.")
    if value is not True:
        raise ConfigError(f"{where}: is mandatory in Config Plan 4A and cannot be false")
    return True


def _unique_texts(
    value: object,
    *,
    where: str,
    allowed: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if type(value) not in (list, tuple) or not value:
        raise ConfigError(f"{where}: must be a non-empty list of unique strings.")
    result: list[str] = []
    for index, item in enumerate(value):
        if not static_isinstance(item, str) or not str.__str__(item):
            raise ConfigError(f"{where}[{index}]: must be a non-empty string.")
        text = str.__str__(item)
        if allowed is not None and text not in allowed:
            raise ConfigError(f"{where}[{index}]: must be one of {list(allowed)}.")
        if text in result:
            raise ConfigError(f"{where}: entries must be unique.")
        result.append(text)
    return tuple(result)


def _product_request(name: str, value: object) -> ProductRequest:
    where = f"outputs.write.{name}"
    if static_isinstance(value, bool):
        if value is not True:
            raise ConfigError(f"{where}: must be true or a mapping.")
        raw: dict[str, object] = {}
    else:
        allowed = ("format", "runs")
        if name in ("aux", "taps"):
            allowed += ("keys",)
        if name == "signal_paths":
            allowed += ("themes",)
        try:
            raw = _closed_mapping(value, where=where, allowed=allowed)
        except ConfigError as error:
            if str(error) == f"{where}: must be a mapping.":
                raise ConfigError(f"{where}: must be true or a mapping.") from None
            raise

    raw_format = raw.get("format", _PRODUCT_DEFAULT_FORMATS[name])
    if (
        not static_isinstance(raw_format, str)
        or str.__str__(raw_format) not in _PRODUCT_FORMATS[name]
    ):
        raise ConfigError(f"{where}.format: must be one of {list(_PRODUCT_FORMATS[name])}.")
    format_ = str.__str__(raw_format)
    runs = ()
    if "runs" in raw:
        runs = _unique_texts(raw["runs"], where=f"{where}.runs")
    options: list[tuple[str, object]] = []
    if "keys" in raw:
        options.append(("keys", _unique_texts(raw["keys"], where=f"{where}.keys")))
    if "themes" in raw:
        options.append(
            (
                "themes",
                _unique_texts(
                    raw["themes"],
                    where=f"{where}.themes",
                    allowed=("light", "dark"),
                ),
            )
        )
    return ProductRequest(name, format_, runs, tuple(options))


def _report_request(value: object) -> ReportRequest:
    where = "outputs.report"
    raw = _closed_mapping(
        value,
        where=where,
        allowed=("rows", "columns", "reference", "relative", "format"),
    )
    if "rows" not in raw:
        raise ConfigError("outputs.report.rows: is required.")
    rows = _unique_texts(raw["rows"], where="outputs.report.rows")
    columns = _unique_texts(
        raw.get("columns", list(_REPORT_COLUMNS)),
        where="outputs.report.columns",
        allowed=_REPORT_COLUMNS,
    )
    reference = raw.get("reference")
    if reference is not None:
        if not static_isinstance(reference, str) or not str.__str__(reference):
            raise ConfigError("outputs.report.reference: must be a non-empty string.")
        reference = str.__str__(reference)
        if reference not in rows:
            raise ConfigError("outputs.report.reference: must name one of outputs.report.rows.")
    relative: tuple[str, ...] = ()
    if "relative" in raw:
        relative = _unique_texts(
            raw["relative"],
            where="outputs.report.relative",
            allowed=_REPORT_RELATIVE,
        )
        if reference is None:
            raise ConfigError("outputs.report.reference: is required for relative columns.")
    raw_formats = raw.get("format", "text")
    if static_isinstance(raw_formats, str):
        formats = _unique_texts(
            [str.__str__(raw_formats)],
            where="outputs.report.format",
            allowed=_REPORT_FORMATS,
        )
    else:
        formats = _unique_texts(
            raw_formats,
            where="outputs.report.format",
            allowed=_REPORT_FORMATS,
        )
    return ReportRequest(rows, columns, cast(str | None, reference), relative, formats)


def parse_output_grammar(raw_outputs: object) -> ParsedOutputSection:
    """Parse and detach only the raw top-level ``outputs`` value."""
    top_allowed = (*_OUTPUT_KEYS, *_PLAN4B_TOP)
    top = _closed_mapping(raw_outputs, where="outputs", allowed=top_allowed)
    for name in _PLAN4B_TOP:
        if name in top:
            raise ConfigError(f"outputs.{name}: capability is reserved for Config Plan 4B")

    directory = top.get("dir")
    if "dir" in top:
        if not static_isinstance(directory, str) or not str.__str__(directory):
            raise ConfigError("outputs.dir: must be a non-empty string.")
        directory = str.__str__(directory)
        if "\0" in directory:
            raise ConfigError("outputs.dir: contains NUL.")

    clobber = top.get("clobber", False)
    if not static_isinstance(clobber, bool):
        raise ConfigError("outputs.clobber: must be a bool.")
    clobber = bool(clobber)

    stdout = top.get("stdout", "summary")
    if not static_isinstance(stdout, str) or str.__str__(stdout) not in _STDOUT:
        raise ConfigError(f"outputs.stdout: must be one of {list(_STDOUT)}.")
    stdout = str.__str__(stdout)

    write = _closed_mapping(
        top.get("write", {}),
        where="outputs.write",
        allowed=(*_WRITE_KEYS, *_PLAN4B_WRITE),
    )
    products = tuple(
        _product_request(name, value) for name, value in write.items() if name in _PLAN4B_WRITE
    )

    write_config = _required_true(write.get("config", True), where="outputs.write.config")
    write_provenance = _required_true(
        write.get("provenance", True),
        where="outputs.write.provenance",
    )
    diagnostics = write.get("diagnostics", True)
    if static_isinstance(diagnostics, bool):
        _required_true(diagnostics, where="outputs.write.diagnostics")
    else:
        diagnostic_mapping = _closed_mapping(
            diagnostics,
            where="outputs.write.diagnostics",
            allowed=("format",),
        )
        if tuple(diagnostic_mapping) != ("format",) or diagnostic_mapping["format"] != "json":
            raise ConfigError("outputs.write.diagnostics.format: must be 'json'.")
    return ParsedOutputSection(
        directory=cast(str | None, directory),
        clobber=clobber,
        stdout=cast(str, stdout),
        write_config=write_config,
        write_provenance=write_provenance,
        write_diagnostics="json",
        products=products,
        report=_report_request(top["report"]) if "report" in top else None,
    )
