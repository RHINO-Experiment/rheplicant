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
    return np.array(array, copy=True, order="C")


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


def _flat_matrix_arrays(value: object, *, name: str) -> dict[str, object]:
    matrix = getattr(value, "matrix", None)
    if matrix is None:
        raise ConfigError(f"{name} product is not a FlatMatrix.")
    return {name: matrix}


def _fisher_arrays(product: object, _configured: object, _options: Mapping[str, object]):
    return _npz(
        {
            **_flat_matrix_arrays(_field(product, "fisher", "fisher"), name="fisher"),
            **_flat_matrix_arrays(
                _field(product, "covariance", "fisher"), name="covariance"
            ),
        }
    )


def _fisher_covariance(product: object, _configured: object, _options: Mapping[str, object]):
    return _npz(
        _flat_matrix_arrays(
            _field(product, "covariance", "fisher"), name="covariance"
        )
    )


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


#: The vocabulary of quality signals a run product may carry.  Deliberately a
#: vocabulary rather than a per-kind field list: the records differ (a NUTS
#: product spells ``divergences``, a GLS result spells ``iterations``/``delta``,
#: a plan nests a ``PlanDiagnostics``), and a per-kind list is the copy that
#: goes stale when a kind gains a field, because nothing renders the two side
#: by side.  A name absent from a product is simply absent from the artefact;
#: nothing is null-filled.
DIAGNOSTIC_FIELDS = (
    "converged",
    "rhat",
    "r_hat",
    "chi2",
    "divergences",
    "n_eff",
    "iterations",
    "delta",
    "rank",
    "nullity",
    "weakest_identified",
    "sweeps",
    "warmup",
    "best_step",
    "n_draw",
    "n_chain",
    "noise_depends_on_prediction",
    "block_residuals",
    "engines",
)


def _diagnostic_json(value: object) -> object:
    """``_json_value``, except that a non-finite number becomes ``null``.

    Everywhere else a NaN in a scientific record is corruption and refusing is
    right.  Here it is a *finding*: numpyro reports ``r_hat`` and ``n_eff`` as
    NaN for a chain that degenerated, and a run diverging on every transition
    is exactly the run whose diagnostics someone needs to read.  Refusing would
    publish nothing for the worst runs, which is backwards, so the undefined
    value is recorded as JSON's own absence-of-a-number rather than dropped.
    """
    if type(value) is float and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _diagnostic_json(value.item())
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        array = _array(value, where="diagnostic record array")
        if array.dtype.kind == "c":
            raise ConfigError("complex arrays belong in NPZ, not a JSON record.")
        return _diagnostic_json(array.tolist())
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_json(child) for key, child in value.items()}
    if type(value) in (tuple, list):
        return [_diagnostic_json(child) for child in value]
    return _json_value(value)


def _diagnostic_lookup(record: object, name: str) -> object:
    """One field of a product record, by key or by attribute, or ``None``.

    Products come in three shapes -- NamedTuple, frozen dataclass, plain dict --
    and one field that matters is a computed property
    (``IdentifiabilityReport.weakest_identified``), invisible to ``vars()`` but
    reachable by ``getattr``.  Reading through both is what lets one extractor
    serve every kind.
    """
    if isinstance(record, Mapping):
        try:
            return record.get(name)
        except Exception:
            return None
    return getattr(record, name, None)


def _run_diagnostics(product: object, _configured: object, _options: Mapping[str, object]):
    """Whatever quality signals this run measured about its own trustworthiness.

    These are the numbers that say whether an answer is what it looks like --
    ``r_hat``, ``divergences``, the joint chi-squared, a conditioning number --
    and until now they lived only on the in-memory product, so a published tree
    recorded that a run happened and not whether to believe it.

    A run whose product carries none of them raises, which the bundle records as
    an omission for that run rather than a failure: asking for diagnostics from
    a forward simulation is not an error, it just has none.
    """
    found: dict[str, object] = {}
    for name in DIAGNOSTIC_FIELDS:
        value = _diagnostic_lookup(product, name)
        if value is not None:
            found[name] = _diagnostic_json(value)
    nested = _diagnostic_lookup(product, "diagnostics")
    if isinstance(nested, Mapping):
        # NUTS nests {latent: {"r_hat": ..., "n_eff": ...}}; keep it per-latent
        # rather than collapsing, because a single bad latent is the finding.
        if nested:
            found["per_latent"] = _diagnostic_json(nested)
    elif nested is not None:
        for name in DIAGNOSTIC_FIELDS:
            if name in found:
                continue
            value = _diagnostic_lookup(nested, name)
            if value is not None:
                found[name] = _diagnostic_json(value)
    if not found:
        raise ConfigError("this run's product carries no diagnostic fields.")
    return ExtractedProduct("json", found)


def _condition_diagnostics(product: object, _configured: object, _options: Mapping[str, object]):
    """``condition``'s product IS the conditioning number, not a record."""
    return ExtractedProduct("json", {"kappa": _diagnostic_json(np.asarray(product))})


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
    """Reduce a ``(n_draw, *grid)`` pushforward across the DRAW axis.

    ``axis=0`` throughout, and that is the whole content of this function: the
    predictions arrive already pushed through the model one draw at a time
    (:func:`~rheplicant.inference.predict_from_samples` vmaps over the draw
    axis), so every statistic here is a reduction OF pushforwards and never the
    model evaluated once at a reduced parameter.  For a nonlinear model the two
    orderings are different numbers, which is the reason
    :func:`~rheplicant.inference.linear.check_linearity` exists at all.

    ``median`` is named for its pair rather than for its quantile.  ``q025``
    and ``q975`` are one thing — the edges of a 95% interval — and slipping a
    ``q050`` between them would read as a third edge of that band.  What the
    median actually pairs with is ``mean``: both are central tendencies, and a
    reader choosing between them is choosing how much a heavy tail should
    move the answer.

    A ``predict`` run reusing a ``fisher`` product never reaches here — it has
    no draw axis to reduce, and :mod:`~rheplicant.config.products.bundle`
    answers it with ``std`` alone.
    """
    array = _array(product, where="prediction")
    if array.ndim == 0:
        raise ConfigError("prediction bands need a prediction with a leading draw axis.")
    return ExtractedProduct(
        "npz",
        {
            "mean": np.mean(array, axis=0),
            "median": np.median(array, axis=0),
            "std": np.std(array, axis=0),
            "q025": np.quantile(array, 0.025, axis=0),
            "q975": np.quantile(array, 0.975, axis=0),
        },
    )


def _wiener_covariance(product: object, _configured: object, _options: Mapping[str, object]):
    return _npz(
        _flat_matrix_arrays(
            _field(product, "covariance", "conjugate.wiener"), name="covariance"
        )
    )


def _wiener_arrays(product: object, _configured: object, _options: Mapping[str, object]):
    mapping = _fixed_mapping(product, "conjugate.wiener")
    values = {"mean": _field(mapping, "mean", "conjugate.wiener")}
    if "covariance" in mapping:
        values["covariance"] = mapping["covariance"].matrix
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


def _compare_record(product: object, _configured: object, _options: Mapping[str, object]):
    return ExtractedProduct("json", dict(product._asdict()))


def _benchmark_record(product: object, _configured: object, _options: Mapping[str, object]):
    variants = []
    for variant in product.variants:
        metrics = {
            name: {
                "samples": list(metric.samples),
                "minimum": metric.minimum,
                "median": metric.median,
                "mean": metric.mean,
                "unit": metric.unit,
            }
            for name, metric in variant.metrics.items()
        }
        variants.append({"name": variant.name, "metrics": metrics})
    return ExtractedProduct("json", {"variants": variants})


RUN_KIND_SELECTORS = {
    "forward": ("arrays", "aux", "taps"),
    "fisher": ("arrays", "covariance"),
    "optimize": ("arrays", "parameters", "losses", "training_history", "run_diagnostics"),
    "plan.estimate": ("arrays", "estimates", "parameters", "recovery", "run_diagnostics"),
    "plan.sample": ("arrays", "draws", "parameters", "chains", "recovery", "run_diagnostics"),
    "conjugate.wiener": (
        "arrays", "estimates", "parameters", "covariance", "recovery", "run_diagnostics",
    ),
    "conjugate.gcr": ("arrays", "draws", "chains", "run_diagnostics"),
    "conjugate.gls": ("arrays", "estimates", "parameters", "recovery", "run_diagnostics"),
    "condition": ("arrays", "run_diagnostics"),
    "identifiability": ("arrays", "identifiability", "run_diagnostics"),
    "score_directions": ("arrays", "scores"),
    "gradient": ("arrays", "gradients"),
    "mmodes": ("arrays",),
    "predict": ("arrays", "prediction_bands", "posterior_predictives"),
    "nuts": ("arrays", "draws", "parameters", "chains", "recovery", "run_diagnostics"),
    "npe": (
        "arrays", "draws", "parameters", "chains", "training_history", "recovery",
        "run_diagnostics",
    ),
    "compare": ("compare",),
    "benchmark": ("benchmark",),
}


_DIAGNOSTIC_KINDS = tuple(
    kind for kind, selectors in RUN_KIND_SELECTORS.items() if "run_diagnostics" in selectors
)

EXTRACTOR_REGISTRY: dict[tuple[str, str], Callable] = {
    ("forward", "arrays"): _forward_arrays,
    ("forward", "aux"): _forward_aux,
    ("forward", "taps"): _forward_taps,
    ("fisher", "arrays"): _fisher_arrays,
    ("fisher", "covariance"): _fisher_covariance,
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
    ("compare", "compare"): _compare_record,
    ("benchmark", "benchmark"): _benchmark_record,
}

# One generic extractor serves every declaring kind, so a kind that gains a
# diagnostic field needs no edit here; `condition` is the exception because its
# product IS the conditioning number rather than a record carrying one.
EXTRACTOR_REGISTRY.update(
    {
        (kind, "run_diagnostics"): (
            _condition_diagnostics if kind == "condition" else _run_diagnostics
        )
        for kind in _DIAGNOSTIC_KINDS
    }
)


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
