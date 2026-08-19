"""Typed extractors for the scientific products of every shipped run kind."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from _rheplicant_bootstrap.audit.names import encode_name
from _rheplicant_bootstrap.errors import ConfigError


@dataclass(frozen=True, slots=True)
class ExtractedProduct:
    encoding: Literal["npz", "json", "txt"]
    value: object
    metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)


def _array(value: object, *, where: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception:
        raise ConfigError(f"{where} is not a numeric array.") from None
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise ConfigError(f"{where} is not a numeric non-object array.")
    return np.ascontiguousarray(array)


def numeric_leaves(value: object) -> dict[str, np.ndarray]:
    """Flatten a known scientific pytree to stable, path-safe numeric keys."""
    found: dict[str, np.ndarray] = {}

    def walk(node: object, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            rows: list[tuple[str, object]] = []
            for raw_name, child in node.items():
                if type(raw_name) is not str or not raw_name:
                    raise ConfigError("numeric mapping keys must be non-empty strings.")
                rows.append((encode_name(raw_name), child))
            for name, child in sorted(rows):
                walk(child, (*path, "mapping", name))
            return
        if type(node) in (tuple, list):
            for index, child in enumerate(node):
                walk(child, (*path, f"i-{index}"))
            return
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for field in dataclasses.fields(node):
                walk(getattr(node, field.name), (*path, "record", field.name))
            return
        key = "/".join(path) or "value"
        if key in found:
            raise ConfigError(f"numeric leaf path {key!r} appears twice.")
        found[key] = _array(node, where=f"numeric leaf {key!r}")

    walk(value, ())
    if not found:
        raise ConfigError("scientific product contains no numeric leaves.")
    return found


def _fixed_mapping(product: object, kind: str) -> Mapping[str, object]:
    if not isinstance(product, Mapping):
        raise ConfigError(f"kind: {kind} returned a non-mapping product.")
    return product


def _field(product: object, field: str, kind: str) -> object:
    mapping = _fixed_mapping(product, kind)
    if field not in mapping:
        raise ConfigError(f"kind: {kind} returned no required {field!r} product field.")
    return mapping[field]


def _npz(value: object) -> ExtractedProduct:
    return ExtractedProduct("npz", numeric_leaves(value))


def _whole_npz(product: object, _configured: object, _options: Mapping[str, object]):
    return _npz(product)


def _forward_arrays(product: object, configured: object, _options: Mapping[str, object]):
    arrays = {"predicted": _array(product.data, where="forward State.data")}
    observed = configured.inference.observed
    if observed is not None:
        for name, value in sorted(observed.entries.items()):
            arrays[f"observed/{encode_name(name)}"] = _array(
                value, where=f"observed {name!r}"
            )
    return ExtractedProduct("npz", arrays)


def _state_aux(product: object, options: Mapping[str, object], *, taps: bool):
    aux = product.aux
    if not isinstance(aux, Mapping):
        raise ConfigError("forward State.aux is not a mapping.")
    requested = options.get("keys")
    if requested is None:
        names = tuple(
            name.removeprefix("snapshot/")
            for name in aux
            if (name.startswith("snapshot/") if taps else True)
        )
    else:
        names = tuple(requested)
    if not names:
        raise ConfigError("requested aux/taps product has no available keys.")
    values: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name in names:
        source = f"snapshot/{name}" if taps and f"snapshot/{name}" in aux else name
        if source not in aux:
            missing.append(name)
            continue
        values[encode_name(name)] = _array(aux[source], where=f"State.aux[{source!r}]")
    if missing:
        raise ConfigError(f"requested aux/taps keys are missing: {missing}.")
    return ExtractedProduct("npz", values)


def _forward_aux(product: object, _configured: object, options: Mapping[str, object]):
    return _state_aux(product, options, taps=False)


def _forward_taps(product: object, _configured: object, options: Mapping[str, object]):
    return _state_aux(product, options, taps=True)


def _mapping_arrays(kind: str, *fields: str) -> Callable:
    def extract(product: object, _configured: object, _options: Mapping[str, object]):
        return _npz({field: _field(product, field, kind) for field in fields})

    return extract


def _mapping_field(kind: str, field: str) -> Callable:
    def extract(product: object, _configured: object, _options: Mapping[str, object]):
        return _npz(_field(product, field, kind))

    return extract


def _attribute_field(field: str) -> Callable:
    def extract(product: object, _configured: object, _options: Mapping[str, object]):
        return _npz(getattr(product, field))

    return extract


def _posterior_parameters(product: object, _configured: object, _options: Mapping[str, object]):
    means = {name: np.mean(_array(stack, where=f"samples[{name!r}]"), axis=0)
             for name, stack in product.samples.items()}
    return _npz(means)


def _losses(product: object, _configured: object, _options: Mapping[str, object]):
    return ExtractedProduct(
        "npz", {"loss": _array(_field(product, "losses", "optimize"), where="losses")}
    )


def _npe_history(product: object, _configured: object, _options: Mapping[str, object]):
    return ExtractedProduct(
        "npz",
        {
            "train": _array(product.train_loss, where="NPE training loss"),
            "validation": _array(product.validation_loss, where="NPE validation loss"),
            "best_step": _array(product.best_step, where="NPE best step"),
        },
    )


def _json_value(value: object) -> object:
    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not np.isfinite(value):
            raise ConfigError("scientific record contains a non-finite number.")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        array = _array(value, where="scientific record array")
        if array.dtype.kind == "c":
            raise ConfigError("complex arrays belong in NPZ, not a JSON record.")
        return _json_value(array.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if type(value) in (tuple, list):
        return [_json_value(child) for child in value]
    raise ConfigError(f"scientific record contains unsupported {type(value).__name__} data.")


def _identifiability(product: object, _configured: object, _options: Mapping[str, object]):
    fields = (
        "names", "shapes", "spans", "n_par", "n_data", "rank", "nullity",
        "singular_values", "null_space", "jacobian", "column_norms", "rtol",
        "threshold", "weakest_identified",
    )
    return ExtractedProduct(
        "json", {field: _json_value(getattr(product, field)) for field in fields}
    )


def _identifiability_arrays(product: object, _configured: object, _options: Mapping[str, object]):
    return _npz(
        {
            "singular_values": product.singular_values,
            "null_space": product.null_space,
            "jacobian": product.jacobian,
            "column_norms": product.column_norms,
        }
    )


def _condition(product: object, _configured: object, _options: Mapping[str, object]):
    return ExtractedProduct("npz", {"condition": _array(product, where="condition")})


def _raw_array(name: str) -> Callable:
    def extract(product: object, _configured: object, _options: Mapping[str, object]):
        return ExtractedProduct("npz", {name: _array(product, where=name)})

    return extract


def _prediction_bands(product: object, _configured: object, _options: Mapping[str, object]):
    array = _array(product, where="prediction")
    if array.ndim == 0:
        raise ConfigError("prediction bands need a prediction with a leading draw axis.")
    return ExtractedProduct(
        "npz",
        {
            "mean": np.mean(array, axis=0),
            "std": np.std(array, axis=0),
            "q025": np.quantile(array, 0.025, axis=0),
            "q975": np.quantile(array, 0.975, axis=0),
        },
    )


def _wiener_covariance(product: object, _configured: object, _options: Mapping[str, object]):
    return _npz(_field(product, "covariance", "conjugate.wiener"))


def _wiener_arrays(product: object, _configured: object, _options: Mapping[str, object]):
    mapping = _fixed_mapping(product, "conjugate.wiener")
    values = {"mean": _field(mapping, "mean", "conjugate.wiener")}
    if "covariance" in mapping:
        values["covariance"] = mapping["covariance"]
    values["residual"] = _field(mapping, "residual", "conjugate.wiener")
    return _npz(values)


def _gcr_arrays(product: object, _configured: object, _options: Mapping[str, object]):
    mapping = _fixed_mapping(product, "conjugate.gcr")
    return _npz({key: mapping[key] for key in ("draws", "residual", "noise_std")})


def _gls_arrays(product: object, _configured: object, _options: Mapping[str, object]):
    return _npz(
        {
            "noise_std": product.noise_std,
            "solution": product.solution,
            "residual": product.residual,
            "iterations": product.iterations,
            "delta": product.delta,
            "converged": product.converged,
        }
    )


def _nuts_arrays(product: object, _configured: object, _options: Mapping[str, object]):
    return _npz(product.samples)


def _npe_arrays(product: object, _configured: object, _options: Mapping[str, object]):
    return _npz(
        {
            "samples": product.samples,
            "train_loss": product.train_loss,
            "validation_loss": product.validation_loss,
            "best_step": product.best_step,
        }
    )


RUN_KIND_SELECTORS = {
    "forward": ("arrays", "aux", "taps"),
    "fisher": ("arrays", "covariance"),
    "optimize": ("arrays", "parameters", "losses", "training_history"),
    "plan.estimate": ("arrays", "estimates", "parameters", "recovery"),
    "plan.sample": ("arrays", "draws", "parameters", "chains", "recovery"),
    "conjugate.wiener": ("arrays", "estimates", "parameters", "covariance", "recovery"),
    "conjugate.gcr": ("arrays", "draws", "chains"),
    "conjugate.gls": ("arrays", "estimates", "parameters", "recovery"),
    "condition": ("arrays",),
    "identifiability": ("arrays", "identifiability"),
    "score_directions": ("arrays", "scores"),
    "gradient": ("arrays", "gradients"),
    "mmodes": ("arrays",),
    "predict": ("arrays", "prediction_bands", "posterior_predictives"),
    "nuts": ("arrays", "draws", "parameters", "chains", "recovery"),
    "npe": ("arrays", "draws", "parameters", "chains", "training_history", "recovery"),
}


EXTRACTOR_REGISTRY: dict[tuple[str, str], Callable] = {
    ("forward", "arrays"): _forward_arrays,
    ("forward", "aux"): _forward_aux,
    ("forward", "taps"): _forward_taps,
    ("fisher", "arrays"): _mapping_arrays("fisher", "fisher", "covariance"),
    ("fisher", "covariance"): _mapping_field("fisher", "covariance"),
    ("optimize", "arrays"): _mapping_arrays("optimize", "params", "losses"),
    ("optimize", "parameters"): _mapping_field("optimize", "params"),
    ("optimize", "losses"): _losses,
    ("optimize", "training_history"): _losses,
    ("plan.estimate", "arrays"): _attribute_field("values"),
    ("plan.estimate", "estimates"): _attribute_field("values"),
    ("plan.estimate", "parameters"): _attribute_field("values"),
    ("plan.sample", "arrays"): _attribute_field("samples"),
    ("plan.sample", "draws"): _attribute_field("samples"),
    ("plan.sample", "parameters"): _posterior_parameters,
    ("plan.sample", "chains"): _attribute_field("samples"),
    ("conjugate.wiener", "arrays"): _wiener_arrays,
    ("conjugate.wiener", "estimates"): _mapping_field("conjugate.wiener", "mean"),
    ("conjugate.wiener", "parameters"): _mapping_field("conjugate.wiener", "mean"),
    ("conjugate.wiener", "covariance"): _wiener_covariance,
    ("conjugate.gcr", "arrays"): _gcr_arrays,
    ("conjugate.gcr", "draws"): _mapping_field("conjugate.gcr", "draws"),
    ("conjugate.gcr", "chains"): _mapping_field("conjugate.gcr", "draws"),
    ("conjugate.gls", "arrays"): _gls_arrays,
    ("conjugate.gls", "estimates"): _attribute_field("solution"),
    ("conjugate.gls", "parameters"): _attribute_field("solution"),
    ("condition", "arrays"): _condition,
    ("identifiability", "arrays"): _identifiability_arrays,
    ("identifiability", "identifiability"): _identifiability,
    ("score_directions", "arrays"): _whole_npz,
    ("score_directions", "scores"): _whole_npz,
    ("gradient", "arrays"): _whole_npz,
    ("gradient", "gradients"): _whole_npz,
    ("mmodes", "arrays"): _raw_array("mmodes"),
    ("predict", "arrays"): _raw_array("prediction"),
    ("predict", "prediction_bands"): _prediction_bands,
    ("predict", "posterior_predictives"): _raw_array("prediction"),
    ("nuts", "arrays"): _nuts_arrays,
    ("nuts", "draws"): _attribute_field("samples"),
    ("nuts", "parameters"): _posterior_parameters,
    ("nuts", "chains"): _attribute_field("samples"),
    ("npe", "arrays"): _npe_arrays,
    ("npe", "draws"): _attribute_field("samples"),
    ("npe", "parameters"): _posterior_parameters,
    ("npe", "chains"): _attribute_field("samples"),
    ("npe", "training_history"): _npe_history,
}


def _recovery(product: object, configured: object, kind: str) -> ExtractedProduct:
    from .recovery import recovery_record

    return ExtractedProduct("json", recovery_record(kind, product, configured.inference))


for _kind in ("plan.estimate", "plan.sample", "conjugate.wiener", "conjugate.gls", "nuts", "npe"):
    EXTRACTOR_REGISTRY[(_kind, "recovery")] = (
        lambda product, configured, _options, kind=_kind: _recovery(product, configured, kind)
    )


def extract_run_payload(
    kind: str,
    selector: str,
    product: object,
    configured: object,
    *,
    options: Sequence[tuple[str, object]] = (),
) -> ExtractedProduct:
    """Extract one semantic product without inspecting destination state."""
    extractor = EXTRACTOR_REGISTRY.get((kind, selector))
    if extractor is None:
        raise ConfigError(f"outputs.write.{selector}: is not compatible with kind: {kind}.")
    return extractor(product, configured, dict(options))


__all__ = [
    "EXTRACTOR_REGISTRY",
    "RUN_KIND_SELECTORS",
    "ExtractedProduct",
    "extract_run_payload",
    "numeric_leaves",
]
